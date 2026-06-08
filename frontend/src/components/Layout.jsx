import Navbar from './Navbar';

export default function Layout({ children, currentPage, onNavigate }) {
  return (
    <div className="app-shell">
      <Navbar currentPage={currentPage} onNavigate={onNavigate} />
      <main className="main-content">{children}</main>
      <footer className="app-footer">
        <p>
          NIT Puducherry — Cardiac Arrest &amp; Heart Risk Prediction System
        </p>
        <p className="footer-disclaimer">
          This is only an ML-based prediction and not a medical diagnosis.
        </p>
      </footer>
    </div>
  );
}
