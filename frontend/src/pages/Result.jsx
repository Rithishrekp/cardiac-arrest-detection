import RiskCard from '../components/RiskCard';

export default function Result({ result, onNavigate }) {
  if (!result) {
    return (
      <div className="page-body">
        <div className="empty-state">
          <span className="empty-icon">🔬</span>
          <h3>No result to display</h3>
          <p>Run a prediction first to see results here.</p>
          <button className="btn btn-primary" onClick={() => onNavigate('predict')}>
            Go to Prediction Form →
          </button>
        </div>
      </div>
    );
  }

  const { record, suggestions, disclaimer } = result;

  return (
    <div className="animate-fadeIn">
      <div className="page-header">
        <h1 className="page-title">Assessment Result</h1>
        <p className="page-subtitle">
          ML-powered cardiac risk prediction for {record.patient_name}
        </p>
      </div>

      <div className="page-body">
        <RiskCard
          record={record}
          suggestions={suggestions}
          disclaimer={disclaimer}
        />

        <div className="result-actions">
          <button className="btn btn-secondary" onClick={() => onNavigate('predict')}>
            ← New Assessment
          </button>
          <button
            className="btn btn-primary"
            onClick={() => onNavigate('history')}
          >
            View History →
          </button>
        </div>
      </div>
    </div>
  );
}
