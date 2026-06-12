const NAV_ITEMS = [
  { id: 'home', label: 'Dashboard', icon: '📊' },
  { id: 'predict', label: 'Predict', icon: '🫀' },
  { id: 'history', label: 'History', icon: '📋' },
  { id: 'report-viewer', label: 'Report Viewer', icon: '📄' },
  { id: 'about', label: 'About', icon: 'ℹ️' },
  { id: 'contact', label: 'Contact', icon: '✉️' },
];

export default function Navbar({ currentPage, onNavigate, user, onLogout }) {
  return (
    <header className="navbar">
      <div className="navbar-brand" onClick={() => onNavigate('home')}>
        <span className="brand-icon">🫀</span>
        <div>
          <div className="brand-title">CardiacRisk AI</div>
        </div>
      </div>

      <div className="navbar-right">
        <nav className="navbar-links">
          {NAV_ITEMS.map(({ id, label, icon }) => (
            <button
              key={id}
              type="button"
              className={`nav-link ${currentPage === id ? 'active' : ''}`}
              onClick={() => onNavigate(id)}
            >
              <span>{icon}</span> {label}
            </button>
          ))}
        </nav>

        {user && (
          <div className="navbar-user">
            <div className="user-avatar" title={user.email}>
              {user.full_name ? user.full_name.charAt(0).toUpperCase() : 'U'}
            </div>
            <span className="user-name">{user.full_name || 'User'}</span>
            <button type="button" className="nav-logout-btn" onClick={onLogout} title="Sign Out">
              <span>🚪</span> Sign Out
            </button>
          </div>
        )}
      </div>
    </header>
  );
}
