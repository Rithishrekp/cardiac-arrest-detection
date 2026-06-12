import { useState } from 'react';
import { api } from '../services/api';

export default function Login({ onAuthSuccess }) {
  const [isRegister, setIsRegister] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    if (!email || !password || (isRegister && !fullName)) {
      setError('Please fill in all fields.');
      setLoading(false);
      return;
    }

    try {
      let data;
      if (isRegister) {
        data = await api.register(email, password, fullName);
      } else {
        data = await api.login(email, password);
      }

      // Store in localStorage
      localStorage.setItem('token', data.token);
      localStorage.setItem('user', JSON.stringify(data.user));

      // Invoke callback
      onAuthSuccess(data.user);
    } catch (err) {
      setError(err.message || 'An error occurred during authentication.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-container">
      {/* Left Panel: Clinical Brand Context */}
      <div className="auth-left-panel">
        <div className="auth-brand-info">
          <div className="auth-logo">
            <span className="logo-symbol">🫀</span>
            <span className="logo-text">CardiacRisk AI</span>
          </div>
          <div className="auth-subtitle">CLINICAL DECISION INTELLIGENCE</div>
        </div>
        
        <div className="auth-hero-content">
          <h1>Professional cardiac arrest risk screening for modern care teams</h1>
          <p>
            An intelligent decision-support system analyzing clinical demographics, sports vitals, 
            and precise ECG intervals. Access comprehensive risk modeling, interactive diagnostics, 
            and professional reporting in a secure environment.
          </p>
        </div>

        <div className="auth-left-footer">
          <span className="security-badge">🛡️ HIPAA Compliance Pathway &amp; Secure Encryption</span>
        </div>
      </div>

      {/* Right Panel: Interactive Card View */}
      <div className="auth-right-panel">
        <div className="auth-card">
          <div className="auth-card-header">
            <h2>{isRegister ? 'Create Clinic Account' : 'Sign In'}</h2>
            <p className="auth-card-desc">
              {isRegister 
                ? 'Register to start running cardiac risk prediction assessments' 
                : 'Enter your clinic credentials to access your workspace'}
            </p>
          </div>

          {error && (
            <div className="auth-error-banner">
              <span className="error-icon">⚠️</span>
              <span className="error-message">{error}</span>
            </div>
          )}

          <form className="auth-form" onSubmit={handleSubmit}>
            {isRegister && (
              <div className="form-group">
                <label htmlFor="fullName">Full Name</label>
                <input
                  type="text"
                  id="fullName"
                  placeholder="Dr. Sarah Jenkins"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  required
                />
              </div>
            )}

            <div className="form-group">
              <label htmlFor="email">Work Email</label>
              <input
                type="email"
                id="email"
                placeholder="jenkins@cardiology.org"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="password">Password</label>
              <input
                type="password"
                id="password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>

            <button type="submit" className="auth-submit-btn" disabled={loading}>
              {loading ? (
                <span className="spinner"></span>
              ) : isRegister ? (
                'Create Account'
              ) : (
                'Sign In'
              )}
            </button>
          </form>

          <div className="auth-card-footer">
            <p>
              {isRegister ? 'Already have an account?' : "Don't have an account?"}{' '}
              <button
                type="button"
                className="auth-toggle-link"
                onClick={() => {
                  setIsRegister(!isRegister);
                  setError('');
                }}
              >
                {isRegister ? 'Sign In' : 'Register Here'}
              </button>
            </p>
            
            <p className="form-disclaimer">
              <strong>Medical Disclaimer:</strong> This software is intended as an early screening tool for clinical decision support. It does not provide direct medical diagnoses. All assessments should be validated by certified medical personnel.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
