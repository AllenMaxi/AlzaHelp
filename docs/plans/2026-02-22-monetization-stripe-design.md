# AlzaHelp Monetization: Stripe Subscription Design

**Date:** 2026-02-22
**Status:** Approved

## Tiers

| | Free | Premium ($9.99/mo) |
|---|---|---|
| Web app (voice, memories, games, meds) | Yes | Yes |
| Push notifications | Yes | Yes |
| Caregiver web dashboard | Yes | Yes |
| External bot (Telegram/WhatsApp) | No | Yes |
| SMS medication alerts | No | Yes |
| Daily AI summaries (bot) | No | Yes |
| Create external bot links | No | Yes |

## Data Model

User document additions (MongoDB `users` collection):

- `subscription_tier: str` — "free" (default) or "premium"
- `stripe_customer_id: Optional[str]` — Stripe customer ID
- `stripe_subscription_id: Optional[str]` — active subscription ID
- `subscription_expires_at: Optional[datetime]` — grace period after cancellation

## Backend Endpoints

| Endpoint | Auth | Purpose |
|---|---|---|
| `POST /api/billing/create-checkout` | JWT | Creates Stripe Checkout session, returns redirect URL |
| `POST /api/billing/create-portal` | JWT | Creates Stripe Customer Portal session (manage/cancel) |
| `POST /api/billing/webhook` | Stripe signature | Handles subscription lifecycle events |
| `GET /api/billing/status` | JWT | Returns current tier + expiration |

### Webhook Events

- `checkout.session.completed` — set tier to "premium", store stripe IDs
- `customer.subscription.updated` — update expiration date
- `customer.subscription.deleted` — set tier to "free" with 3-day grace period
- `invoice.payment_failed` — send push notification to user

## Feature Gating

Helper function `require_premium(user)` raises HTTP 403 if user is not premium.

Applied to:
- `POST /api/external-bot-links` (creating bot links)
- `POST /webhooks/telegram/bot` (bot message processing)
- `POST /webhooks/whatsapp/bot` (bot message processing)
- `send_sms_fallback()` (SMS alerts)

Free users can still access the caregiver portal web UI — gating only applies to external bot and SMS features.

## Frontend Changes

1. **Upgrade banner** in CaregiverPortalSection — shown when free user tries to create bot link
2. **Subscription status** in DashboardPage settings section — current plan + "Manage Subscription" button
3. **Pricing cards** on LandingPage — 2-column Free vs Premium comparison

## Payment Flow

1. User clicks "Upgrade to Premium"
2. Frontend calls `POST /api/billing/create-checkout`
3. Backend creates Stripe Checkout session with `mode: "subscription"`, returns URL
4. User redirected to Stripe-hosted checkout
5. User pays → Stripe fires `checkout.session.completed` webhook
6. Backend webhook handler updates user's `subscription_tier` to "premium"
7. User redirected back to app with `?session_id=...` success URL

## Subscription Management

- "Manage Subscription" button calls `POST /api/billing/create-portal`
- Redirects to Stripe-hosted Customer Portal
- User can update payment method, cancel, view invoices
- Cancellation triggers `customer.subscription.deleted` webhook → 3-day grace → tier reset

## Environment Variables

```
STRIPE_SECRET_KEY=sk_...
STRIPE_WEBHOOK_SECRET=whsec_...
STRIPE_PRICE_ID=price_...
```

## Implementation Approach

- Use Stripe Checkout (redirect) — no custom payment UI needed
- Use Stripe Customer Portal — no subscription management UI needed
- Minimal frontend: just upgrade buttons + status display
- All payment logic handled by Stripe webhooks
