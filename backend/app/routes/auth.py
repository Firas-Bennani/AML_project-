import uuid
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from slowapi import Limiter
from slowapi.util import get_remote_address
from app.database import get_db
from app.dependencies import required_role
from app.models.user import User, UserRole
from app.schemas.user import UserCreate, UserLogin, UserOut, TokenOut
from app.auth import hash_password, verify_password, create_access_token
from app.audit import log_action
from app.config import MAX_LOGIN_ATTEMPTS

limiter = Limiter(key_func=get_remote_address)


router = APIRouter(prefix="/auth", tags=["Authetication"])


@router.post("/register", response_model=UserOut, status_code=201)
def register(payload: UserCreate, db: Session = Depends(get_db)):

    existing_user = db.query(User).filter(User.email == payload.email).first()

    if existing_user:
        raise HTTPException(
            status_code=400, detail="A user with this email already exists"
        )

    hashed = hash_password(payload.password)

    new_user = User(
        id=str(uuid.uuid4()),
        name=payload.name,
        email=payload.email,
        password_hash=hashed,
        role=payload.role,
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    log_action(
        db=db,
        action="REGISTER",
        user_id=new_user.id,
        entity_type="USER",
        entity_id=new_user.id,
        details=f"New user registered with role {new_user.role.value}",
    )

    return new_user


@router.post("/login", response_model=TokenOut)
@limiter.limit("10/minute")
def login(request: Request, payload: UserLogin, db: Session = Depends(get_db)):
    
    user = db.query(User).filter(User.email == payload.email).first()
    
    if user and user.failed_attempts >= MAX_LOGIN_ATTEMPTS:
        raise HTTPException(
            status_code=423,
            detail="Acount locked due to many failed attempts.Contact an Admin.",
        )

    if not user or not verify_password(payload.password, user.password_hash):
        if user:
            user.failed_attempts += 1
            db.commit()

            log_action(
                db=db,
                action="LOGIN_FAILED",
                user_id=user.id,
                entity_type="USER",
                entity_id=user.id,
                details=f"Failed login attempt {user.failed_attempts}/5 for {user.email}",
            )

        raise HTTPException(status_code=401, detail="Incorrect email or password")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="This account has been disabled")

    user.failed_attempts = 0  # reset failed attempts counter after successful login
    db.commit()

    token = create_access_token(user_id=user.id, role=user.role)

    log_action(
        db=db,
        action="LOGIN",
        user_id=user.id,
        entity_type="USER",
        entity_id=user.id,
        details=f"Successful login by {user.email}",
    )

    return {"access_token": token, "token_type": "bearer"}


@router.post("/unlock/{user_id}", response_model=UserOut)
def unlock_account(
    user_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(required_role([UserRole.ADMIN])),
):

    target_user = db.query(User).filter(User.id == user_id).first()

    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")

    target_user.failed_attempts = 0
    target_user.is_active = True

    db.commit()
    db.refresh(target_user)

    log_action(
        db=db,
        action="UNLOCK_ACCOUNT",
        user_id=current_user.id,
        entity_type="USER",
        entity_id=target_user.id,
        details=f"Account unlocked by admin {current_user.email}",
    )

    return target_user
