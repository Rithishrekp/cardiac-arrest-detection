import { useState } from 'react';

export default function Contact() {
  const [form, setForm] = useState({ name: '', email: '', subject: 'Research Inquiry', message: '' });
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState(null);

  function handleChange(e) {
    const { name, value } = e.target;
    setForm(prev => ({ ...prev, [name]: value }));
  }

  function handleSubmit(e) {
    e.preventDefault();
    if (!form.name || !form.email || !form.message) {
      setError('Please fill in all required fields.');
      return;
    }
    setError(null);
    setSubmitted(true);
    setForm({ name: '', email: '', subject: 'Research Inquiry', message: '' });
  }

  return (
    <div className="animate-fadeIn">
      <div className="page-header">
        <h1 className="page-title">Contact Us</h1>
        <p className="page-subtitle">Get in touch with the Sudden Cardiac Arrest AI Research Group</p>
      </div>

      <div className="page-body" style={{ display: 'grid', gridTemplateColumns: '3fr 2fr', gap: '24px' }}>
        {/* Contact Form Card */}
        <div className="card">
          <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: '18px', fontWeight: 700, marginBottom: '16px' }}>
            📬 Send a Message
          </h2>
          
          {submitted ? (
            <div className="alert normal" style={{ background: 'rgba(16,185,129,0.12)', borderColor: 'var(--risk-normal)', padding: '20px', borderRadius: '8px' }}>
              <span style={{ fontSize: '24px', marginRight: '12px' }}>✅</span>
              <div>
                <strong>Thank you!</strong>
                <p style={{ margin: '4px 0 0 0', fontSize: '13px', color: 'var(--text-secondary)' }}>
                  Your inquiry has been successfully sent. A research team representative will get back to you shortly.
                </p>
              </div>
              <button className="btn btn-primary btn-sm" style={{ marginTop: '12px' }} onClick={() => setSubmitted(false)}>
                Send Another Message
              </button>
            </div>
          ) : (
            <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              {error && (
                <div className="alert danger">
                  <span>⚠️</span>
                  <div>{error}</div>
                </div>
              )}

              <div className="form-field">
                <label htmlFor="contact-name">Full Name <span className="required">*</span></label>
                <input
                  id="contact-name"
                  type="text"
                  name="name"
                  value={form.name}
                  onChange={handleChange}
                  placeholder="e.g. Dr. Rajesh Sharma"
                  required
                />
              </div>

              <div className="form-field">
                <label htmlFor="contact-email">Email Address <span className="required">*</span></label>
                <input
                  id="contact-email"
                  type="email"
                  name="email"
                  value={form.email}
                  onChange={handleChange}
                  placeholder="e.g. rajesh@hospital.org"
                  required
                />
              </div>

              <div className="form-field">
                <label htmlFor="contact-subject">Inquiry Type</label>
                <select id="contact-subject" name="subject" value={form.subject} onChange={handleChange}>
                  <option value="Research Inquiry">Research & Collaboration Inquiry</option>
                  <option value="Technical Support">Technical Support</option>
                  <option value="Report Verification">Patient Report Verification</option>
                  <option value="Feedback">System Feedback</option>
                </select>
              </div>

              <div className="form-field">
                <label htmlFor="contact-message">Message <span className="required">*</span></label>
                <textarea
                  id="contact-message"
                  name="message"
                  rows={5}
                  value={form.message}
                  onChange={handleChange}
                  placeholder="Provide detailed description of your request or observation..."
                  required
                  style={{
                    width: '100%',
                    padding: '10px 14px',
                    borderRadius: 'var(--radius-md)',
                    border: '1px solid var(--border)',
                    background: 'var(--bg-glass-input)',
                    color: 'var(--text-primary)',
                    fontFamily: 'inherit',
                    resize: 'vertical'
                  }}
                />
              </div>

              <button type="submit" className="btn btn-primary" style={{ alignSelf: 'flex-start', padding: '10px 24px' }}>
                ✉️ Send Message
              </button>
            </form>
          )}
        </div>

        {/* Contact info Sidebar */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <div className="card">
            <h3 style={{ fontFamily: 'var(--font-heading)', fontSize: '16px', fontWeight: 700, marginBottom: '12px' }}>
              🏥 Research Lab Info
            </h3>
            <ul className="info-list" style={{ fontSize: '13px', lineHeight: '1.6' }}>
              <li>
                <strong>Division:</strong><br />
                Department of Machine Learning and Healthcare Technologies
              </li>
              <li>
                <strong>Project Domain:</strong><br />
                Sports Physiology and Arrhythmia Detection Algorithms
              </li>
              <li>
                <strong>Emails:</strong><br />
                research-sca@cardiac-ai.org<br />
                techsupport@cardiac-ai.org
              </li>
            </ul>
          </div>

          <div className="disclaimer-banner" style={{ margin: 0, padding: '16px', borderRadius: 'var(--radius-lg)' }}>
            <span>⚕️</span>
            <div>
              <strong style={{ fontSize: '13px' }}>AI Screening Warning</strong>
              <p style={{ fontSize: '11px', lineHeight: '1.4', margin: '4px 0 0 0' }}>
                This system is an AI-assisted early sudden cardiac arrest risk screening platform intended for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis or treatment.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
