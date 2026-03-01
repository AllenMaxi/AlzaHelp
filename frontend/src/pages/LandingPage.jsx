import React from "react";
import { useNavigate, Link } from "react-router-dom";
import { useTranslation } from 'react-i18next';
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Mic, Pill, Users, Shield, Heart, ArrowRight, Brain, MessageCircle, Check, Crown } from "lucide-react";
import { useAuth } from "@/context/AuthContext";

export const LandingPage = () => {
  const navigate = useNavigate();
  const { t, i18n } = useTranslation();
  const { demoLogin } = useAuth();

  const features = [
    {
      icon: Mic,
      title: t('landing.features.voice.title'),
      description: t('landing.features.voice.desc'),
    },
    {
      icon: Pill,
      title: t('landing.features.medication.title'),
      description: t('landing.features.medication.desc'),
    },
    {
      icon: Users,
      title: t('landing.features.family.title'),
      description: t('landing.features.family.desc'),
    },
    {
      icon: Shield,
      title: t('landing.features.safety.title'),
      description: t('landing.features.safety.desc'),
    },
  ];

  const roles = [
    {
      icon: Heart,
      title: t('landing.roles.patients.title'),
      description: t('landing.roles.patients.desc'),
    },
    {
      icon: MessageCircle,
      title: t('landing.roles.caregivers.title'),
      description: t('landing.roles.caregivers.desc'),
    },
    {
      icon: Brain,
      title: t('landing.roles.ai.title'),
      description: t('landing.roles.ai.desc'),
    },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-b from-violet-50 via-white to-violet-50/30 dark:from-gray-950 dark:via-gray-900 dark:to-gray-950">
      {/* Language Toggle */}
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 pt-4 flex justify-end">
        <div className="flex gap-1 text-sm">
          <button
            onClick={() => i18n.changeLanguage('en')}
            className={`px-2 py-1 rounded ${i18n.language?.startsWith('en') ? 'bg-violet-100 dark:bg-violet-900/50 text-violet-700 dark:text-violet-300 font-medium' : 'text-gray-500 hover:text-gray-700'}`}
          >
            EN
          </button>
          <button
            onClick={() => i18n.changeLanguage('es')}
            className={`px-2 py-1 rounded ${i18n.language?.startsWith('es') ? 'bg-violet-100 dark:bg-violet-900/50 text-violet-700 dark:text-violet-300 font-medium' : 'text-gray-500 hover:text-gray-700'}`}
          >
            ES
          </button>
        </div>
      </div>

      {/* Hero */}
      <header className="container mx-auto px-4 sm:px-6 lg:px-8 pt-16 pb-20 text-center">
        <div className="inline-flex items-center gap-2 bg-violet-100 dark:bg-violet-900/30 text-violet-700 dark:text-violet-300 rounded-full px-4 py-1.5 text-sm font-medium mb-8">
          <Heart className="h-4 w-4" fill="currentColor" />
          {t('landing.badge')}
        </div>
        <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight text-gray-900 dark:text-white leading-tight">
          {t('landing.title')}{" "}
          <span className="text-violet-600 dark:text-violet-400">
            {t('landing.titleHighlight')}
          </span>
        </h1>
        <p className="mt-6 text-lg sm:text-xl text-gray-600 dark:text-gray-300 max-w-2xl mx-auto leading-relaxed">
          {t('landing.subtitle')}
        </p>
        <div className="mt-10 flex flex-col sm:flex-row gap-4 justify-center">
          <Button
            size="lg"
            className="text-lg px-8 py-6"
            onClick={() => navigate("/login")}
          >
            {t('landing.getStarted')}
            <ArrowRight className="ml-2 h-5 w-5" />
          </Button>
          <Button
            size="lg"
            variant="outline"
            className="text-lg px-8 py-6"
            onClick={() => navigate("/login")}
          >
            {t('landing.learnMore')}
          </Button>
          <Button
            variant="outline"
            size="lg"
            className="text-lg px-8 py-6 border-violet-300 text-violet-700 dark:text-violet-300 hover:bg-violet-50 dark:hover:bg-violet-900/30"
            onClick={async () => {
              try {
                await demoLogin();
                navigate('/dashboard');
              } catch (err) {
                console.error('Demo failed:', err);
              }
            }}
          >
            {t('landing.tryDemo')}
          </Button>
        </div>
      </header>

      {/* Features */}
      <section className="container mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <h2 className="text-3xl font-bold text-center mb-12 text-gray-900 dark:text-white">
          Everything you need, in one place
        </h2>
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {features.map(({ icon: Icon, title, description }) => (
            <Card
              key={title}
              className="border-2 hover:border-violet-300 dark:hover:border-violet-700 transition-colors"
            >
              <CardHeader>
                <div className="h-12 w-12 rounded-xl bg-violet-100 dark:bg-violet-900/30 flex items-center justify-center mb-3">
                  <Icon className="h-6 w-6 text-violet-600 dark:text-violet-400" />
                </div>
                <CardTitle className="text-lg">{title}</CardTitle>
                <CardDescription className="text-sm leading-relaxed">
                  {description}
                </CardDescription>
              </CardHeader>
            </Card>
          ))}
        </div>
      </section>

      {/* How It Works */}
      <section className="bg-violet-600 dark:bg-violet-900 text-white py-16">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-center mb-12">
            Built for every member of the care team
          </h2>
          <div className="grid gap-8 sm:grid-cols-3 max-w-4xl mx-auto">
            {roles.map(({ icon: Icon, title, description }) => (
              <div key={title} className="text-center">
                <div className="h-14 w-14 rounded-2xl bg-white/20 flex items-center justify-center mx-auto mb-4">
                  <Icon className="h-7 w-7" />
                </div>
                <h3 className="text-xl font-semibold mb-2">{title}</h3>
                <p className="text-violet-100 text-sm leading-relaxed">
                  {description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Pricing */}
      <section className="container mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <h2 className="text-3xl font-bold text-center text-gray-900 dark:text-white mb-4">
          {t('landing.pricing.title')}
        </h2>
        <p className="text-center text-gray-600 dark:text-gray-300 mb-12 max-w-xl mx-auto">
          The patient app is always free. Premium features for caregivers who need remote access.
        </p>
        <div className="grid gap-8 sm:grid-cols-2 max-w-3xl mx-auto">
          {/* Free Tier */}
          <Card className="border-2 border-gray-200 dark:border-gray-700 relative">
            <CardHeader className="pb-2">
              <CardTitle className="text-xl">{t('landing.pricing.free.name')}</CardTitle>
              <CardDescription>For patients & families</CardDescription>
            </CardHeader>
            <div className="px-6 pb-6">
              <p className="text-4xl font-bold text-gray-900 dark:text-white mb-6">{t('landing.pricing.free.price')}<span className="text-base font-normal text-gray-500">/{t('landing.pricing.free.period')}</span></p>
              <ul className="space-y-3 mb-6">
                {(t('landing.pricing.free.features', { returnObjects: true }) || []).map(f => (
                  <li key={f} className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300">
                    <Check className="h-4 w-4 text-green-500 flex-shrink-0" />
                    {f}
                  </li>
                ))}
              </ul>
              <Button variant="outline" className="w-full" onClick={() => navigate("/login")}>
                Get Started
              </Button>
            </div>
          </Card>

          {/* Premium Tier */}
          <Card className="border-2 border-violet-500 relative">
            <div className="absolute -top-3 left-1/2 -translate-x-1/2">
              <span className="inline-flex items-center gap-1 bg-violet-600 text-white text-xs font-semibold px-3 py-1 rounded-full">
                <Crown className="h-3 w-3" /> Most Popular
              </span>
            </div>
            <CardHeader className="pb-2">
              <CardTitle className="text-xl">{t('landing.pricing.premium.name')}</CardTitle>
              <CardDescription>For caregivers & clinicians</CardDescription>
            </CardHeader>
            <div className="px-6 pb-6">
              <p className="text-4xl font-bold text-gray-900 dark:text-white mb-6">{t('landing.pricing.premium.price')}<span className="text-base font-normal text-gray-500">/{t('landing.pricing.premium.period')}</span></p>
              <ul className="space-y-3 mb-6">
                {(t('landing.pricing.premium.features', { returnObjects: true }) || []).map(f => (
                  <li key={f} className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300">
                    <Check className="h-4 w-4 text-violet-500 flex-shrink-0" />
                    {f}
                  </li>
                ))}
              </ul>
              <Button className="w-full bg-violet-600 hover:bg-violet-700 text-white" onClick={() => navigate("/login")}>
                Start Free, Upgrade Anytime
              </Button>
            </div>
          </Card>
        </div>
      </section>

      {/* CTA */}
      <section className="container mx-auto px-4 sm:px-6 lg:px-8 py-20 text-center">
        <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
          {t('landing.cta')}
        </h2>
        <p className="text-gray-600 dark:text-gray-300 mb-8 max-w-xl mx-auto">
          Free to use. Set up in under a minute. No credit card required.
        </p>
        <Button
          size="lg"
          className="text-lg px-8 py-6"
          onClick={() => navigate("/login")}
        >
          {t('landing.ctaButton')}
          <ArrowRight className="ml-2 h-5 w-5" />
        </Button>
      </section>

      {/* Footer */}
      <footer className="border-t border-gray-200 dark:border-gray-800 py-8">
        <div className="container mx-auto px-4 text-center text-sm text-gray-500 dark:text-gray-400">
          <div className="flex items-center justify-center gap-2 mb-2">
            <Heart className="h-4 w-4 text-violet-500" fill="currentColor" />
            <span className="font-medium">AlzaHelp</span>
          </div>
          <p>Made with care for those who need it most.</p>
          <div className="mt-3 flex items-center justify-center gap-4 text-xs">
            <Link to="/privacy" className="hover:text-foreground transition-colors">Privacy Policy</Link>
            <span>·</span>
            <Link to="/terms" className="hover:text-foreground transition-colors">Terms of Service</Link>
          </div>
        </div>
      </footer>
    </div>
  );
};
