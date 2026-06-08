import { useState } from 'react';
import Layout from './components/Layout';
import Home from './pages/Home';
import Predict from './pages/Predict';
import Result from './pages/Result';
import History from './pages/History';
import About from './pages/About';

export default function App() {
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
      case 'about':
        return <About />;
      default:
        return <Home onNavigate={navigate} />;
    }
  }

  return (
    <Layout currentPage={page} onNavigate={navigate}>
      {renderPage()}
    </Layout>
  );
}
