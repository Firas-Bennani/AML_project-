import { NavLink, useNavigate } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';

// ── Nav Item ──────────────────────────────────────────────────
function NavItem({ to, icon, label }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `flex items-center gap-3 px-4 py-2.5 rounded-lg text-sm font-medium transition-colors ${
          isActive
            ? 'bg-green-500/10 text-green-400 border border-green-500/20'
            : 'text-gray-400 hover:text-white hover:bg-gray-800 border border-transparent'
        }`
      }
    >
      <span className="text-base">{icon}</span>
      <span>{label}</span>
    </NavLink>
  );
}

// ── Sidebar ───────────────────────────────────────────────────
function Sidebar() {
  const { user, logout } = useAuthStore();
  const navigate = useNavigate();

  const isPrivileged = user?.role === 'ADMIN' || user?.role === 'AUDITOR';

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const getRoleColor = (role) => {
    switch (role) {
      case 'ADMIN':
        return 'text-red-400 bg-red-500/10 border-red-500/20';
      case 'AUDITOR':
        return 'text-blue-400 bg-blue-500/10 border-blue-500/20';
      case 'ANALYST':
        return 'text-green-400 bg-green-500/10 border-green-500/20';
      default:
        return 'text-gray-400 bg-gray-500/10 border-gray-500/20';
    }
  };

  return (
    <aside className="fixed top-0 left-0 h-screen w-56 bg-gray-900 border-r border-gray-800 flex flex-col z-40">
      {/* Logo */}
      <div className="px-5 py-6 border-b border-gray-800">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-green-500/10 border border-green-500/20 rounded-lg flex items-center justify-center text-sm">
            🛡️
          </div>
          <div>
            <p className="text-white text-sm font-semibold leading-none">AML Platform</p>
            <p className="text-gray-600 text-xs mt-0.5">Monitoring System</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-5 space-y-1 overflow-y-auto">
        <p className="text-gray-600 text-xs uppercase tracking-widest px-4 mb-3">Main</p>
        <NavItem to="/dashboard" icon="📊" label="Dashboard" />
        <NavItem to="/transactions" icon="💳" label="Transactions" />
        <NavItem to="/alerts" icon="🔔" label="Alerts" />

        {/* Reports — only visible to ADMIN and AUDITOR */}
        {isPrivileged && (
          <>
            <p className="text-gray-600 text-xs uppercase tracking-widest px-4 mt-6 mb-3">
              Analytics
            </p>
            <NavItem to="/reports" icon="📈" label="Reports" />
          </>
        )}
      </nav>

      {/* User Info + Logout */}
      <div className="px-3 py-4 border-t border-gray-800">
        {/* User Card */}
        <div className="px-3 py-3 mb-3 bg-gray-800/50 rounded-lg">
          <p className="text-white text-xs font-medium truncate">{user?.name}</p>
          <p className="text-gray-500 text-xs truncate mt-0.5">{user?.email}</p>
          <span
            className={`inline-block mt-2 text-xs px-2 py-0.5 rounded-full border font-medium ${getRoleColor(user?.role)}`}
          >
            {user?.role}
          </span>
        </div>

        {/* Logout Button */}
        <button
          onClick={handleLogout}
          className="w-full flex items-center gap-3 px-4 py-2.5 rounded-lg text-sm font-medium text-gray-400 hover:text-red-400 hover:bg-red-500/5 border border-transparent hover:border-red-500/20 transition-colors"
        >
          <span>🚪</span>
          <span>Sign Out</span>
        </button>
      </div>
    </aside>
  );
}

export default Sidebar;
