import Navbar from './Navbar';

export default function Layout({ children, currentPage, onNavigate, user, onLogout }) {
  return (
    <div className="app-shell">
      <Navbar currentPage={currentPage} onNavigate={onNavigate} user={user} onLogout={onLogout} />
      <main className="main-content">{children}</main>
      <footer className="app-footer">
        <p>
          Cardiac Arrest &amp; Heart Risk Prediction System
        </p>
        <p className="footer-disclaimer">
          This is only an ML-based prediction and not a medical diagnosis.
        </p>
      </footer>
    </div>
  );
}
