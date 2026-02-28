import React from "react";
import { Link } from "react-router-dom";
import { ArrowLeft } from "lucide-react";

export const PrivacyPolicyPage = () => (
  <div className="min-h-screen bg-background">
    <div className="container mx-auto px-4 py-12 max-w-3xl">
      <Link to="/" className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground mb-8">
        <ArrowLeft className="h-4 w-4" /> Back to home
      </Link>
      <h1 className="text-3xl font-bold mb-2">Privacy Policy</h1>
      <p className="text-muted-foreground mb-8">Last updated: February 23, 2026</p>

      <div className="prose prose-gray dark:prose-invert max-w-none space-y-6">
        <section>
          <h2 className="text-xl font-semibold mt-6 mb-3">1. Information We Collect</h2>
          <p className="text-muted-foreground leading-relaxed">
            AlzaHelp collects information you provide directly: account details (name, email), health data (medications, care instructions, mood observations), family information, memories, and voice recordings when using the voice assistant. We also collect usage data (pages visited, features used) through PostHog analytics.
          </p>
        </section>

        <section>
          <h2 className="text-xl font-semibold mt-6 mb-3">2. How We Use Your Information</h2>
          <p className="text-muted-foreground leading-relaxed">
            We use your data to provide and improve the AlzaHelp service: managing medications, delivering care instructions via voice assistant, enabling caregiver access, sending reminders and push notifications, and generating AI-powered responses. Health data is processed by OpenAI's API for voice and chat features.
          </p>
        </section>

        <section>
          <h2 className="text-xl font-semibold mt-6 mb-3">3. Data Sharing</h2>
          <p className="text-muted-foreground leading-relaxed">
            We do not sell your personal data. Data is shared only with: caregivers/clinicians you explicitly authorize, third-party services required for functionality (OpenAI for AI features, Stripe for payments, Twilio for SMS alerts), and as required by law.
          </p>
        </section>

        <section>
          <h2 className="text-xl font-semibold mt-6 mb-3">4. Data Security</h2>
          <p className="text-muted-foreground leading-relaxed">
            We use encryption in transit (TLS/HTTPS), secure password hashing (bcrypt), JWT-based authentication, and role-based access controls. Data is stored in MongoDB with access restricted to authorized services.
          </p>
        </section>

        <section>
          <h2 className="text-xl font-semibold mt-6 mb-3">5. Your Rights</h2>
          <p className="text-muted-foreground leading-relaxed">
            You can export all your data at any time from your account settings. You can delete your account and all associated data permanently. Caregivers can only access data you explicitly share via invite codes.
          </p>
        </section>

        <section>
          <h2 className="text-xl font-semibold mt-6 mb-3">6. Data Retention</h2>
          <p className="text-muted-foreground leading-relaxed">
            We retain your data for as long as your account is active. When you delete your account, all personal data is permanently removed within 30 days. Session data expires automatically after 30 days of inactivity.
          </p>
        </section>

        <section>
          <h2 className="text-xl font-semibold mt-6 mb-3">7. Contact</h2>
          <p className="text-muted-foreground leading-relaxed">
            For privacy questions or data requests, contact us at privacy@alzahelp.com.
          </p>
        </section>
      </div>
    </div>
  </div>
);

export default PrivacyPolicyPage;
