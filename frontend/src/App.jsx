import { useState, useEffect } from 'react';
import Layout from './components/Layout';
import Home from './pages/Home';
import Predict from './pages/Predict';
import Result from './pages/Result';
import History from './pages/History';
import About from './pages/About';
import ReportViewer from './pages/ReportViewer';
import Contact from './pages/Contact';
import Login from './pages/Login';

export default function App() {
  const [user, setUser] = useState(() => {
    const saved = localStorage.getItem('user');
    try {
      return saved ? JSON.parse(saved) : null;
    } catch {
      return null;
    }
  });
  const [page, setPage] = useState('home');
  const [lastResult, setLastResult] = useState(null);

  function navigate(target) {
    setPage(target);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function handleResult(result) {
    setLastResult(result);
    navigate('result');
  }

  function handleAuthSuccess(authenticatedUser) {
    setUser(authenticatedUser);
    setPage('home');
  }

  function handleLogout() {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setUser(null);
    setPage('home');
  }

  if (!user) {
    return <Login onAuthSuccess={handleAuthSuccess} />;
  }

  function renderPage() {
    switch (page) {
      case 'home':
        return <Home onNavigate={navigate} />;
      case 'predict':
        return <Predict onResult={handleResult} />;
      case 'result':
        return <Result result={lastResult} onNavigate={navigate} />;
      case 'history':
        return <History onNavigate={navigate} />;
      case 'report-viewer':
        return <ReportViewer onNavigate={navigate} />;
      case 'about':
        return <About />;
      case 'contact':
        return <Contact />;
      default:
        return <Home onNavigate={navigate} />;
    }
  }

  return (
    <Layout currentPage={page} onNavigate={navigate} user={user} onLogout={handleLogout}>
      {renderPage()}
    </Layout>
  );
}
