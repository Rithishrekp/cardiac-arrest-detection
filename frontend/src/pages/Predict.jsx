import { useState } from 'react';
import { api } from '../services/api';

const INITIAL_FORM = {
  patient_id: '',
  patient_name: '',
  age: '',
  gender: 'Male',
  weight: '',
  height: '',
  heart_rate: '',
  systolic_bp: '',
  diastolic_bp: '',
  sport_type: 'ATH',
  rr_interval: '',
  pp_interval: '',
  qt_interval: '',
  qtc_interval: '',
  qrs_duration: '',
  pq_interval: '',
  family_history_heart_disease: false,
  personal_history_heart_disease: false,
  syncope: false,
  pectus_excavatum: false,
};

const SPORT_TYPES = [
  { value: 'ATH', label: 'Athletics / Running' },
  { value: 'AMF', label: 'American Football' },
  { value: 'VOL', label: 'Volleyball' },
  { value: 'CYC', label: 'Cycling' },
  { value: 'RUN', label: 'Running / Sprinting' },
  { value: 'TRE', label: 'Trekking / Climbing' },
  { value: 'FUT', label: 'Football / Soccer' },
  { value: 'CFIT', label: 'CrossFit / Powerlifting' },
];

function validate(form) {
  const errors = {};

  if (!form.patient_id.trim()) errors.patient_id = 'Patient ID is required';
  if (!form.patient_name.trim()) errors.patient_name = 'Patient name is required';

  const age = Number(form.age);
  if (!form.age || isNaN(age) || age < 1 || age > 120) {
    errors.age = 'Age must be between 1 and 120';
  }

  const weight = Number(form.weight);
  if (!form.weight || isNaN(weight) || weight < 5 || weight > 300) {
    errors.weight = 'Weight must be between 5 and 300 kg';
  }

  const height = Number(form.height);
  if (!form.height || isNaN(height) || height < 50 || height > 250) {
    errors.height = 'Height must be between 50 and 250 cm';
  }

  const hr = Number(form.heart_rate);
  if (!form.heart_rate || isNaN(hr) || hr < 30 || hr > 220) {
    errors.heart_rate = 'Heart rate must be 30–220 bpm';
  }

  const sbp = Number(form.systolic_bp);
  if (!form.systolic_bp || isNaN(sbp) || sbp < 50 || sbp > 250) {
    errors.systolic_bp = 'Systolic BP must be 50–250 mmHg';
  }

  const dbp = Number(form.diastolic_bp);
  if (!form.diastolic_bp || isNaN(dbp) || dbp < 35 || dbp > 160) {
    errors.diastolic_bp = 'Diastolic BP must be 35–160 mmHg';
  }

  const rr = Number(form.rr_interval);
  if (!form.rr_interval || isNaN(rr) || rr < 100 || rr > 2000) {
    errors.rr_interval = 'RR interval must be 100–2000 ms';
  }

  const pp = Number(form.pp_interval);
  if (!form.pp_interval || isNaN(pp) || pp < 100 || pp > 2000) {
    errors.pp_interval = 'PP interval must be 100–2000 ms';
  }

  const qt = Number(form.qt_interval);
  if (!form.qt_interval || isNaN(qt) || qt < 100 || qt > 800) {
    errors.qt_interval = 'QT interval must be 100–800 ms';
  }

  const qtc = Number(form.qtc_interval);
  if (!form.qtc_interval || isNaN(qtc) || qtc < 100 || qtc > 800) {
    errors.qtc_interval = 'QTc interval must be 100–800 ms';
  }

  const qrs = Number(form.qrs_duration);
  if (!form.qrs_duration || isNaN(qrs) || qrs < 20 || qrs > 250) {
    errors.qrs_duration = 'QRS duration must be 20–250 ms';
  }

  const pq = Number(form.pq_interval);
  if (!form.pq_interval || isNaN(pq) || pq < 20 || pq > 400) {
    errors.pq_interval = 'PQ interval must be 20–400 ms';
  }

  return errors;
}

export default function Predict({ onResult }) {
  const [form, setForm] = useState(INITIAL_FORM);
  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);
  const [apiError, setApiError] = useState(null);
  const [activeTab, setActiveTab] = useState('demographics');

  function handleChange(e) {
    const { name, value, type, checked } = e.target;
    setForm((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }));
    setErrors((prev) => ({ ...prev, [name]: undefined }));
    setApiError(null);
  }

  function fillSample() {
    setForm({
      patient_id: 'P-2024-ATH',
      patient_name: 'Vikram Singh',
      age: '26',
      gender: 'Male',
      weight: '74',
      height: '180',
      heart_rate: '68',
      systolic_bp: '124',
      diastolic_bp: '82',
      sport_type: 'ATH',
      rr_interval: '880',
      pp_interval: '875',
      qt_interval: '390',
      qtc_interval: '415',
      qrs_duration: '98',
      pq_interval: '155',
      family_history_heart_disease: true,
      personal_history_heart_disease: false,
      syncope: false,
      pectus_excavatum: false,
    });
    setErrors({});
    setApiError(null);
  }

  async function handleSubmit(e) {
    e.preventDefault();
    const validationErrors = validate(form);
    if (Object.keys(validationErrors).length > 0) {
      setErrors(validationErrors);
      // Automatically switch tab to where the first error is
      if (validationErrors.patient_id || validationErrors.patient_name || validationErrors.age || validationErrors.weight || validationErrors.height) {
        setActiveTab('demographics');
      } else if (validationErrors.rr_interval || validationErrors.pp_interval || validationErrors.qt_interval || validationErrors.qtc_interval || validationErrors.qrs_duration || validationErrors.pq_interval || validationErrors.heart_rate || validationErrors.systolic_bp || validationErrors.diastolic_bp) {
        setActiveTab('vitals');
      }
      return;
    }

    setLoading(true);
    setApiError(null);

    try {
      const payload = {
        patient_id: form.patient_id.trim(),
        patient_name: form.patient_name.trim(),
        age: Number(form.age),
        gender: form.gender,
        weight: Number(form.weight),
        height: Number(form.height),
        heart_rate: Number(form.heart_rate),
        systolic_bp: Number(form.systolic_bp),
        diastolic_bp: Number(form.diastolic_bp),
        sport_type: form.sport_type,
        rr_interval: Number(form.rr_interval),
        pp_interval: Number(form.pp_interval),
        qt_interval: Number(form.qt_interval),
        qtc_interval: Number(form.qtc_interval),
        qrs_duration: Number(form.qrs_duration),
        pq_interval: Number(form.pq_interval),
        family_history_heart_disease: form.family_history_heart_disease ? 1.0 : 0.0,
        personal_history_heart_disease: form.personal_history_heart_disease ? 1.0 : 0.0,
        syncope: form.syncope ? 1.0 : 0.0,
        pectus_excavatum: form.pectus_excavatum ? 1.0 : 0.0,
      };

      const result = await api.predict(payload);
      onResult(result);
    } catch (err) {
      setApiError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="animate-fadeIn">
      <div className="page-header">
        <h1 className="page-title">Cardiac Assessment Form</h1>
        <p className="page-subtitle">
          Submit full demographic, physiological, and ECG vital signs to predict cardiac arrest risk.
        </p>
      </div>

      <div className="page-body">
        <div className="form-layout">
          <form className="card prediction-form" onSubmit={handleSubmit} noValidate>
            
            {/* Form Steps / Tab Navigation */}
            <div className="form-tabs" style={{ display: 'flex', borderBottom: '1px solid var(--border)', marginBottom: '1.5rem' }}>
              <button
                type="button"
                className={`tab-btn ${activeTab === 'demographics' ? 'active' : ''}`}
                onClick={() => setActiveTab('demographics')}
                style={{ padding: '0.75rem 1.25rem', border: 'none', background: 'none', borderBottom: activeTab === 'demographics' ? '3px solid var(--primary)' : 'none', cursor: 'pointer', fontWeight: '600' }}
              >
                👤 Demographics & Sport
              </button>
              <button
                type="button"
                className={`tab-btn ${activeTab === 'history' ? 'active' : ''}`}
                onClick={() => setActiveTab('history')}
                style={{ padding: '0.75rem 1.25rem', border: 'none', background: 'none', borderBottom: activeTab === 'history' ? '3px solid var(--primary)' : 'none', cursor: 'pointer', fontWeight: '600' }}
              >
                📋 Medical History
              </button>
              <button
                type="button"
                className={`tab-btn ${activeTab === 'vitals' ? 'active' : ''}`}
                onClick={() => setActiveTab('vitals')}
                style={{ padding: '0.75rem 1.25rem', border: 'none', background: 'none', borderBottom: activeTab === 'vitals' ? '3px solid var(--primary)' : 'none', cursor: 'pointer', fontWeight: '600' }}
              >
                ⚡ Vitals & ECG Intervals
              </button>
            </div>

            {/* TAB 1: Demographics */}
            {activeTab === 'demographics' && (
              <div className="form-section animate-fadeIn">
                <h3 className="form-section-title">Patient Profile</h3>
                <div className="form-grid">
                  <FormField
                    label="Patient ID / MRN"
                    name="patient_id"
                    value={form.patient_id}
                    onChange={handleChange}
                    error={errors.patient_id}
                    placeholder="e.g. P-1001"
                    required
                  />
                  <FormField
                    label="Full Name"
                    name="patient_name"
                    value={form.patient_name}
                    onChange={handleChange}
                    error={errors.patient_name}
                    placeholder="e.g. Vikram Singh"
                    required
                  />
                  <FormField
                    label="Age (years)"
                    name="age"
                    type="number"
                    value={form.age}
                    onChange={handleChange}
                    error={errors.age}
                    placeholder="1–120"
                    required
                  />
                  <div className="form-field">
                    <label htmlFor="gender">Gender</label>
                    <select id="gender" name="gender" value={form.gender} onChange={handleChange}>
                      <option value="Male">Male</option>
                      <option value="Female">Female</option>
                      <option value="Other">Other</option>
                    </select>
                  </div>
                  <FormField
                    label="Weight (kg)"
                    name="weight"
                    type="number"
                    value={form.weight}
                    onChange={handleChange}
                    error={errors.weight}
                    placeholder="5–300"
                    required
                  />
                  <FormField
                    label="Height (cm)"
                    name="height"
                    type="number"
                    value={form.height}
                    onChange={handleChange}
                    error={errors.height}
                    placeholder="50–250"
                    required
                  />
                  <div className="form-field">
                    <label htmlFor="sport_type">Sport Category</label>
                    <select id="sport_type" name="sport_type" value={form.sport_type} onChange={handleChange}>
                      {SPORT_TYPES.map((s) => (
                        <option key={s.value} value={s.value}>
                          {s.label}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>
            )}

            {/* TAB 2: Medical History */}
            {activeTab === 'history' && (
              <div className="form-section animate-fadeIn">
                <h3 className="form-section-title">Anamnesis Details</h3>
                <p className="form-hint">Tick the cardiac and genetic history conditions that apply to the athlete.</p>
                
                <div className="checkbox-list" style={{ display: 'grid', gap: '1rem', marginTop: '1rem' }}>
                  <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      name="family_history_heart_disease"
                      checked={form.family_history_heart_disease}
                      onChange={handleChange}
                      style={{ width: '1.25rem', height: '1.25rem' }}
                    />
                    <div>
                      <strong>Family History of Heart Disease</strong>
                      <div className="form-hint" style={{ margin: 0 }}>Genetic predisposition to heart attacks or arrest.</div>
                    </div>
                  </label>

                  <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      name="personal_history_heart_disease"
                      checked={form.personal_history_heart_disease}
                      onChange={handleChange}
                      style={{ width: '1.25rem', height: '1.25rem' }}
                    />
                    <div>
                      <strong>Personal History of Heart Disease</strong>
                      <div className="form-hint" style={{ margin: 0 }}>Prior diagnosed arrhythmias, valve issues, or heart conditions.</div>
                    </div>
                  </label>

                  <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      name="syncope"
                      checked={form.syncope}
                      onChange={handleChange}
                      style={{ width: '1.25rem', height: '1.25rem' }}
                    />
                    <div>
                      <strong>Syncope / Unexplained Fainting</strong>
                      <div className="form-hint" style={{ margin: 0 }}>History of sudden fainting spells, especially during exertion.</div>
                    </div>
                  </label>

                  <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      name="pectus_excavatum"
                      checked={form.pectus_excavatum}
                      onChange={handleChange}
                      style={{ width: '1.25rem', height: '1.25rem' }}
                    />
                    <div>
                      <strong>Pectus Excavatum (Sunken Chest)</strong>
                      <div className="form-hint" style={{ margin: 0 }}>Skeletal structural issue compressing the chest cavities.</div>
                    </div>
                  </label>
                </div>
              </div>
            )}

            {/* TAB 3: Vitals & ECG Intervals */}
            {activeTab === 'vitals' && (
              <div className="form-section animate-fadeIn">
                <h3 className="form-section-title">Physiological Vitals & ECG Measurements</h3>
                <div className="form-grid">
                  <FormField
                    label="Heart Rate (bpm)"
                    name="heart_rate"
                    type="number"
                    value={form.heart_rate}
                    onChange={handleChange}
                    error={errors.heart_rate}
                    placeholder="30–220"
                    required
                  />
                  <FormField
                    label="Systolic BP (mmHg)"
                    name="systolic_bp"
                    type="number"
                    value={form.systolic_bp}
                    onChange={handleChange}
                    error={errors.systolic_bp}
                    placeholder="e.g. 120"
                    required
                  />
                  <FormField
                    label="Diastolic BP (mmHg)"
                    name="diastolic_bp"
                    type="number"
                    value={form.diastolic_bp}
                    onChange={handleChange}
                    error={errors.diastolic_bp}
                    placeholder="e.g. 80"
                    required
                  />
                  <FormField
                    label="RR Interval (ms)"
                    name="rr_interval"
                    type="number"
                    value={form.rr_interval}
                    onChange={handleChange}
                    error={errors.rr_interval}
                    placeholder="e.g. 880"
                    required
                  />
                  <FormField
                    label="PP Interval (ms)"
                    name="pp_interval"
                    type="number"
                    value={form.pp_interval}
                    onChange={handleChange}
                    error={errors.pp_interval}
                    placeholder="e.g. 875"
                    required
                  />
                  <FormField
                    label="QT Interval (ms)"
                    name="qt_interval"
                    type="number"
                    value={form.qt_interval}
                    onChange={handleChange}
                    error={errors.qt_interval}
                    placeholder="e.g. 390"
                    required
                  />
                  <FormField
                    label="QTc Interval (ms)"
                    name="qtc_interval"
                    type="number"
                    value={form.qtc_interval}
                    onChange={handleChange}
                    error={errors.qtc_interval}
                    placeholder="e.g. 415"
                    required
                  />
                  <FormField
                    label="QRS Duration (ms)"
                    name="qrs_duration"
                    type="number"
                    value={form.qrs_duration}
                    onChange={handleChange}
                    error={errors.qrs_duration}
                    placeholder="e.g. 98"
                    required
                  />
                  <FormField
                    label="PQ Interval (ms)"
                    name="pq_interval"
                    type="number"
                    value={form.pq_interval}
                    onChange={handleChange}
                    error={errors.pq_interval}
                    placeholder="e.g. 155"
                    required
                  />
                </div>
              </div>
            )}

            {apiError && (
              <div className="alert danger" style={{ marginTop: '1.5rem' }}>
                <span className="alert-icon">⚠️</span>
                <div>
                  <strong>Prediction failed</strong>
                  <br />
                  {apiError}
                </div>
              </div>
            )}

            <div className="form-actions" style={{ marginTop: '2rem' }}>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={fillSample}
                disabled={loading}
              >
                Fill Sample Athlete Vitals
              </button>
              <button type="submit" className="btn btn-primary btn-lg" disabled={loading}>
                {loading ? (
                  <>
                    <span className="spinner spinner-sm" /> Running Risk Classifier…
                  </>
                ) : (
                  '⚡ Predict Risk Score'
                )}
              </button>
            </div>
          </form>

          <aside className="form-sidebar">
            <div className="card info-card">
              <h3>Scientific Features</h3>
              <ul className="info-list">
                <li><strong>Sport Type</strong> - Athlete sports category shifts base cardiac expectations.</li>
                <li><strong>ECG Intervals</strong> - Depolarization and repolarization values (QTc, RR, PP).</li>
                <li><strong>History Flags</strong> - Syncope and family genetic parameters have strong weight in XGBoost.</li>
              </ul>
            </div>

            <div className="card info-card">
              <h3>Diagnostic Guidelines</h3>
              <div className="risk-legend">
                <div className="legend-item normal">
                  <span>0–40%</span> Low Risk
                </div>
                <div className="legend-item medium">
                  <span>41–70%</span> Moderate Risk
                </div>
                <div className="legend-item critical">
                  <span>71–100%</span> High Risk / Alert
                </div>
              </div>
            </div>
          </aside>
        </div>
      </div>
    </div>
  );
}

function FormField({
  label,
  name,
  value,
  onChange,
  error,
  type = 'text',
  placeholder,
  required,
  min,
  max,
  step,
}) {
  return (
    <div className="form-field">
      <label htmlFor={name}>
        {label}
        {required && <span className="required">*</span>}
      </label>
      <input
        id={name}
        name={name}
        type={type}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        className={error ? 'input-error' : ''}
        min={min}
        max={max}
        step={step}
      />
      {error && <span className="field-error">{error}</span>}
    </div>
  );
}
