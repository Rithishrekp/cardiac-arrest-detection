const NAV_ITEMS = [
  { id: 'home', label: 'Dashboard', icon: '📊' },
  { id: 'predict', label: 'Predict', icon: '🫀' },
  { id: 'history', label: 'History', icon: '📋' },
  { id: 'about', label: 'About', icon: 'ℹ️' },
];

export default function Navbar({ currentPage, onNavigate }) {
  return (
    <header className="navbar">
      <div className="navbar-brand" onClick={() => onNavigate('home')}>
        <span className="brand-icon">🫀</span>
        <div>
          <div className="brand-title">CardiacRisk AI</div>
        </div>
      </div>

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
    </header>
  );
}
