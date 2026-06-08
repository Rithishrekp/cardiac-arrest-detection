import { useState } from 'react';
import { api } from '../services/api';

const INITIAL_FORM = {
  patient_id: '',
  patient_name: '',
  age: '',
  gender: 'Male',
  rr_interval: '',
  pp_interval: '',
  qt_interval: '',
};

function validate(form) {
  const errors = {};

  if (!form.patient_id.trim()) errors.patient_id = 'Patient ID is required';
  if (!form.patient_name.trim()) errors.patient_name = 'Patient name is required';

  const age = Number(form.age);
  if (!form.age || isNaN(age) || age < 1 || age > 120) {
    errors.age = 'Age must be between 1 and 120';
  }

  if (!form.gender) errors.gender = 'Gender is required';

  const rr = Number(form.rr_interval);
  if (!form.rr_interval || isNaN(rr) || rr < 200 || rr > 2000) {
    errors.rr_interval = 'RR interval must be 200–2000 ms';
  }

  const pp = Number(form.pp_interval);
  if (!form.pp_interval || isNaN(pp) || pp < 200 || pp > 2000) {
    errors.pp_interval = 'PP interval must be 200–2000 ms';
  }

  const qt = Number(form.qt_interval);
  if (!form.qt_interval || isNaN(qt) || qt < 200 || qt > 800) {
    errors.qt_interval = 'QT interval must be 200–800 ms';
  }

  return errors;
}

export default function Predict({ onResult }) {
  const [form, setForm] = useState(INITIAL_FORM);
  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);
  const [apiError, setApiError] = useState(null);

  function handleChange(e) {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: value }));
    setErrors((prev) => ({ ...prev, [name]: undefined }));
    setApiError(null);
  }

  function fillSample() {
    setForm({
      patient_id: 'P-1001',
      patient_name: 'Ramesh Kumar',
      age: '45',
      gender: 'Male',
      rr_interval: '1084',
      pp_interval: '1090',
      qt_interval: '448',
    });
    setErrors({});
    setApiError(null);
  }

  async function handleSubmit(e) {
    e.preventDefault();
    const validationErrors = validate(form);
    if (Object.keys(validationErrors).length > 0) {
      setErrors(validationErrors);
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
        rr_interval: Number(form.rr_interval),
        pp_interval: Number(form.pp_interval),
        qt_interval: Number(form.qt_interval),
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
        <h1 className="page-title">Cardiac Risk Assessment</h1>
        <p className="page-subtitle">
          Enter patient demographics and ECG interval measurements for ML prediction
        </p>
      </div>

      <div className="page-body">
        <div className="form-layout">
          <form className="card prediction-form" onSubmit={handleSubmit} noValidate>
            <div className="form-section">
              <h3 className="form-section-title">Patient Information</h3>
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
                  placeholder="e.g. Ramesh Kumar"
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
                  min={1}
                  max={120}
                  required
                />
                <div className="form-field">
                  <label htmlFor="gender">Gender</label>
                  <select
                    id="gender"
                    name="gender"
                    value={form.gender}
                    onChange={handleChange}
                    className={errors.gender ? 'input-error' : ''}
                  >
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                    <option value="Other">Other</option>
                  </select>
                  {errors.gender && <span className="field-error">{errors.gender}</span>}
                </div>
              </div>
            </div>

            <div className="form-section">
              <h3 className="form-section-title">ECG Interval Measurements</h3>
              <p className="form-hint">
                The ML model uses RR, PP, and QT intervals (in milliseconds). Rolling
                statistics from the last 5 readings per patient improve accuracy.
              </p>
              <div className="form-grid">
                <FormField
                  label="RR Interval (ms)"
                  name="rr_interval"
                  type="number"
                  value={form.rr_interval}
                  onChange={handleChange}
                  error={errors.rr_interval}
                  placeholder="e.g. 1084"
                  step="0.1"
                  required
                />
                <FormField
                  label="PP Interval (ms)"
                  name="pp_interval"
                  type="number"
                  value={form.pp_interval}
                  onChange={handleChange}
                  error={errors.pp_interval}
                  placeholder="e.g. 1090"
                  step="0.1"
                  required
                />
                <FormField
                  label="QT Interval (ms)"
                  name="qt_interval"
                  type="number"
                  value={form.qt_interval}
                  onChange={handleChange}
                  error={errors.qt_interval}
                  placeholder="e.g. 448"
                  step="0.1"
                  required
                />
              </div>
            </div>

            {apiError && (
              <div className="alert danger">
                <span className="alert-icon">⚠️</span>
                <div>
                  <strong>Prediction failed</strong>
                  <br />
                  {apiError}
                </div>
              </div>
            )}

            <div className="form-actions">
              <button
                type="button"
                className="btn btn-secondary"
                onClick={fillSample}
                disabled={loading}
              >
                Fill Sample Data
              </button>
              <button type="submit" className="btn btn-primary btn-lg" disabled={loading}>
                {loading ? (
                  <>
                    <span className="spinner spinner-sm" /> Analyzing…
                  </>
                ) : (
                  '⚡ Run Prediction'
                )}
              </button>
            </div>
          </form>

          <aside className="form-sidebar">
            <div className="card info-card">
              <h3>About ECG Intervals</h3>
              <ul className="info-list">
                <li><strong>RR Interval</strong> — Time between consecutive R-peaks (heart rate)</li>
                <li><strong>PP Interval</strong> — Time between consecutive P-waves (atrial activity)</li>
                <li><strong>QT Interval</strong> — Ventricular depolarization + repolarization duration</li>
              </ul>
              <p className="info-note">
                QT ≥ 450 ms is a key indicator of elevated cardiac risk in this model.
              </p>
            </div>

            <div className="card info-card">
              <h3>Risk Categories</h3>
              <div className="risk-legend">
                <div className="legend-item normal">
                  <span>0–40%</span> Low / Normal
                </div>
                <div className="legend-item medium">
                  <span>41–70%</span> Medium Risk
                </div>
                <div className="legend-item critical">
                  <span>71–100%</span> High / Critical
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
