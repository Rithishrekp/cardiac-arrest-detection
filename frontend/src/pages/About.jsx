export default function About() {
  return (
    <div className="animate-fadeIn">
      <div className="page-header">
        <h1 className="page-title">About This Project</h1>
        <p className="page-subtitle">
          Internship project — National Institute of Technology, Puducherry
        </p>
      </div>

      <div className="page-body about-layout">
        <div className="card">
          <h2>Problem Statement</h2>
          <p>
            Cardiac arrest and heart rhythm abnormalities are leading causes of mortality
            worldwide. Early detection of elevated cardiac risk from ECG interval data can
            help clinicians intervene before a critical event. This project builds an
            ML-powered web application that predicts cardiac arrest / heart risk from
            patient ECG interval measurements.
          </p>
        </div>

        <div className="card">
          <h2>How It Works</h2>
          <ol className="about-list">
            <li>Patient demographics and ECG intervals (RR, PP, QT in ms) are submitted via the web form.</li>
            <li>The backend computes 18 sliding-window features from the patient&apos;s reading history.</li>
            <li>A trained <strong>XGBoost</strong> classifier predicts the probability of high cardiac risk.</li>
            <li>The risk score (0–100%) is mapped to Low / Medium / High categories with health suggestions.</li>
            <li>Every assessment is stored in the database for history and dashboard analytics.</li>
          </ol>
        </div>

        <div className="about-grid">
          <div className="card">
            <h3>ML Model</h3>
            <ul className="info-list">
              <li><strong>Algorithm:</strong> XGBoost (Gradient Boosting)</li>
              <li><strong>Features:</strong> 18 ECG sliding-window statistics</li>
              <li><strong>Inputs:</strong> RR, PP, QT intervals (milliseconds)</li>
              <li><strong>Window:</strong> Last 5 readings per patient</li>
              <li><strong>Output:</strong> Risk probability → 0–100 score</li>
            </ul>
          </div>

          <div className="card">
            <h3>Tech Stack</h3>
            <ul className="info-list">
              <li><strong>Frontend:</strong> React.js + Vite</li>
              <li><strong>Backend:</strong> FastAPI (Python)</li>
              <li><strong>Database:</strong> SQLite (local) / PostgreSQL (production)</li>
              <li><strong>ML:</strong> scikit-learn, XGBoost, pandas</li>
            </ul>
          </div>
        </div>

        <div className="card">
          <h2>API Endpoints</h2>
          <div className="table-wrapper">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Method</th>
                  <th>Endpoint</th>
                  <th>Description</th>
                </tr>
              </thead>
              <tbody>
                <tr><td className="mono">GET</td><td className="mono">/health</td><td>API and ML engine health check</td></tr>
                <tr><td className="mono">POST</td><td className="mono">/predict</td><td>Submit patient data, get prediction</td></tr>
                <tr><td className="mono">GET</td><td className="mono">/history</td><td>List all prediction records</td></tr>
                <tr><td className="mono">GET</td><td className="mono">/history/stats</td><td>Dashboard statistics</td></tr>
                <tr><td className="mono">GET</td><td className="mono">/history/&#123;id&#125;</td><td>Single record with suggestions</td></tr>
              </tbody>
            </table>
          </div>
        </div>

        <div className="card">
          <h2>Future Enhancements</h2>
          <ul className="info-list">
            <li>Real-time ECG waveform upload and CNN-BiLSTM ensemble inference</li>
            <li>User authentication and role-based access for clinicians</li>
            <li>PostgreSQL deployment with Docker Compose</li>
            <li>PDF report export for each assessment</li>
            <li>Live monitoring dashboard with WebSocket streaming</li>
            <li>Model retraining pipeline with new clinical data</li>
          </ul>
        </div>

        <div className="disclaimer-banner">
          <span>⚕️</span>
          <div>
            <strong>Medical Disclaimer</strong>
            <p>
              This application provides ML-based cardiac risk predictions for research and
              educational purposes only. It is NOT a substitute for professional medical
              diagnosis, treatment, or advice. Always consult a qualified healthcare provider
              for medical decisions.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
