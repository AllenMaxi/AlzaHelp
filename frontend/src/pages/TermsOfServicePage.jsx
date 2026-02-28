import React from "react";
import { Link } from "react-router-dom";
import { ArrowLeft } from "lucide-react";

export const TermsOfServicePage = () => (
  <div className="min-h-screen bg-background">
    <div className="container mx-auto px-4 py-12 max-w-3xl">
      <Link to="/" className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground mb-8">
        <ArrowLeft className="h-4 w-4" /> Back to home
      </Link>
      <h1 className="text-3xl font-bold mb-2">Terms of Service</h1>
      <p className="text-muted-foreground mb-8">Last updated: February 23, 2026</p>

      <div className="prose prose-gray dark:prose-invert max-w-none space-y-6">
        <section>
          <h2 className="text-xl font-semibold mt-6 mb-3">1. Acceptance of Terms</h2>
          <p className="text-muted-foreground leading-relaxed">
            By using AlzaHelp, you agree to these terms. If you do not agree, do not use the service. AlzaHelp is designed to assist with memory care management but is not a substitute for professional medical advice, diagnosis, or treatment.
          </p>
        </section>

        <section>
          <h2 className="text-xl font-semibold mt-6 mb-3">2. Service Description</h2>
          <p className="text-muted-foreground leading-relaxed">
            AlzaHelp provides memory assistance tools including medication tracking, voice-guided care instructions, caregiver coordination, safety monitoring, and AI-powered companionship. The service is available as a web application with optional Telegram/WhatsApp bot integration.
          </p>
        </section>

        <section>
          <h2 className="text-xl font-semibold mt-6 mb-3">3. Account Responsibilities</h2>
          <p className="text-muted-foreground leading-relaxed">
            You are responsible for maintaining the security of your account. Caregivers and clinicians must only access patient data through proper authorization (invite codes). You must not share account credentials or invite codes with unauthorized persons.
          </p>
        </section>

        <section>
          <h2 className="text-xl font-semibold mt-6 mb-3">4. Subscription and Billing</h2>
          <p className="text-muted-foreground leading-relaxed">
            AlzaHelp offers a free tier (up to 3 patients) and a premium subscription ($9.99/month, up to 20 patients with external bot access). Subscriptions are managed through Stripe. You may cancel at any time through the billing portal.
          </p>
        </section>

        <section>
          <h2 className="text-xl font-semibold mt-6 mb-3">5. Medical Disclaimer</h2>
          <p className="text-muted-foreground leading-relaxed">
            AlzaHelp is a care coordination tool, not a medical device. Medication reminders and care instructions are informational only. Always consult healthcare professionals for medical decisions. AlzaHelp is not responsible for missed medications or care actions.
          </p>
        </section>

        <section>
          <h2 className="text-xl font-semibold mt-6 mb-3">6. Limitation of Liability</h2>
          <p className="text-muted-foreground leading-relaxed">
            AlzaHelp is provided "as is" without warranties. We are not liable for any damages arising from use of the service, including but not limited to missed alerts, data loss, or service interruptions. Our total liability is limited to the amount you paid in the 12 months preceding the claim.
          </p>
        </section>

        <section>
          <h2 className="text-xl font-semibold mt-6 mb-3">7. Termination</h2>
          <p className="text-muted-foreground leading-relaxed">
            You may delete your account at any time. We reserve the right to suspend accounts that violate these terms or engage in abusive behavior. Upon termination, your data will be deleted per our privacy policy.
          </p>
        </section>

        <section>
          <h2 className="text-xl font-semibold mt-6 mb-3">8. Contact</h2>
          <p className="text-muted-foreground leading-relaxed">
            For questions about these terms, contact us at legal@alzahelp.com.
          </p>
        </section>
      </div>
    </div>
  </div>
);

export default TermsOfServicePage;
