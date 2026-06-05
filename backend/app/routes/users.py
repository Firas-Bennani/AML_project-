from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.dependencies import required_role
from app.models.user import User, UserRole
from app.schemas.user import UserOut

router = APIRouter(prefix="/users", tags=["Users"])

@router.get("/me", response_model=UserOut)
def get_current_user(
    current_user: User = Depends(required_role([UserRole.ADMIN, UserRole.ANALYST, UserRole.AUDITOR]))
):
    return current_user