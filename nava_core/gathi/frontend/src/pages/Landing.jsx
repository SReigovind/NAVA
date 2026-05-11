import React from "react";
import { useNavigate } from "react-router-dom";

export default function Landing() {
  const navigate = useNavigate();

  return (
    <div className="landing-new">
      <header className="landing-header-new">
        <div className="landing-logo-new">
          <img src="/api/logo" alt="NAVA Logo" />
          <span>NAVA</span>
        </div>
        <button className="btn btn-primary btn-sm" onClick={() => navigate("/auth")}>Log In</button>
      </header>

      <main className="landing-main-new">
        <section className="hero-section-new">
          <div className="hero-content-new">
            <div className="hero-badge">Next-gen Agricultural Virtual Assistant</div>
            <h1 className="hero-title-new">Digital Agronomy for<br/>Every Farmer.</h1>
            <p className="hero-desc-new">
              Empowering agriculture through advanced AI. Detect crop diseases instantly, monitor physiological stress before symptoms appear, and receive expert guidance directly from your smartphone.
            </p>
            <div className="hero-actions-new">
              <button className="btn btn-primary btn-lg" onClick={() => navigate("/auth")}>Get Started</button>
            </div>
          </div>
          <div className="hero-visual-new">
            <div className="bento-grid">
              <div className="bento-card bento-main">
                <div className="bento-icon">🔬</div>
                <h3>Disease Detection</h3>
                <p>Identify 34 diseases across 7 crops with high precision and visual Grad-CAM explanations.</p>
              </div>
              <div className="bento-card bento-sub1">
                <div className="bento-icon">📡</div>
                <h3>Stress Monitoring</h3>
                <p>Detect plant stress before visible symptoms appear using VNIR estimations.</p>
              </div>
              <div className="bento-card bento-sub2">
                <div className="bento-icon">💬</div>
                <h3>Expert AI Chat</h3>
                <p>Multilingual, context-aware advice tailored to your specific field.</p>
              </div>
            </div>
          </div>
        </section>
      </main>

      <footer className="landing-footer-new">
        <div className="footer-content-new">
          <div className="footer-left">
            <div className="footer-brand">
              <img src="/api/logo" alt="NAVA" />
              <span>NAVA</span>
            </div>
            <p className="footer-tagline">Bridging traditional farming and advanced artificial intelligence to secure global agricultural yields.</p>
          </div>
          <div className="footer-right">
            <h4>Project Information</h4>
            <p><strong>Degree:</strong> M.Sc. Artificial Intelligence and Machine Learning (2024–2026)</p>
            <p><strong>Institution:</strong> School of Artificial Intelligence and Robotics, Mahatma Gandhi University, Kottayam, Kerala</p>
            <p><strong>Team:</strong> Dhanus VS · Sreegovind S</p>
          </div>
        </div>
        <div className="footer-bottom-new">
          <p>© 2026 NAVA Project. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}
