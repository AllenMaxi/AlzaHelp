import React, { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Heart, Pill, Users, ArrowRight, X, Check, Camera, Download } from "lucide-react";
import { medicationsApi, familyApi, uploadApi } from "@/services/api";
import { toast } from "sonner";
import { useTranslation } from "react-i18next";

export const OnboardingWizard = ({ onComplete, userName = "Friend" }) => {
  const { t } = useTranslation();
  const [step, setStep] = useState(0);
  const [medName, setMedName] = useState("");
  const [medDosage, setMedDosage] = useState("");
  const [familyName, setFamilyName] = useState("");
  const [familyRelationship, setFamilyRelationship] = useState("");
  const [saving, setSaving] = useState(false);
  const [photoUploaded, setPhotoUploaded] = useState(false);
  const [deferredPrompt, setDeferredPrompt] = useState(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    const handler = (e) => {
      e.preventDefault();
      setDeferredPrompt(e);
    };
    window.addEventListener('beforeinstallprompt', handler);
    return () => window.removeEventListener('beforeinstallprompt', handler);
  }, []);

  const totalSteps = 5;

  const finish = () => {
    localStorage.setItem("alzahelp_onboarded", "true");
    onComplete();
  };

  const handlePhotoUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setSaving(true);
    try {
      await uploadApi.uploadFile(file);
      setPhotoUploaded(true);
      toast.success(t('common.success'));
      setTimeout(() => setStep(2), 500);
    } catch (err) {
      toast.error(t('common.error'));
    } finally {
      setSaving(false);
    }
  };

  const saveMedication = async () => {
    if (!medName.trim()) { toast.error(t('common.error')); return; }
    setSaving(true);
    try {
      await medicationsApi.create({
        name: medName.trim(),
        dosage: medDosage.trim() || "as prescribed",
        frequency: "daily",
        times_per_day: 1,
        scheduled_times: ["08:00"],
        active: true,
      });
      toast.success(`${medName} added!`);
      setStep(3);
    } catch (e) {
      toast.error(t('common.error'));
    } finally {
      setSaving(false);
    }
  };

  const saveFamilyMember = async () => {
    if (!familyName.trim()) { toast.error(t('common.error')); return; }
    setSaving(true);
    try {
      await familyApi.create({
        name: familyName.trim(),
        relationship: familyRelationship.trim() || "family",
        relationship_label: familyRelationship.trim() || "Family Member",
        category: "other",
      });
      toast.success(`${familyName} added!`);
      setStep(4);
    } catch (e) {
      toast.error(t('common.error'));
    } finally {
      setSaving(false);
    }
  };

  const handleInstall = async () => {
    if (!deferredPrompt) {
      finish();
      return;
    }
    deferredPrompt.prompt();
    const { outcome } = await deferredPrompt.userChoice;
    if (outcome === 'accepted') {
      localStorage.setItem('alzahelp_installed', 'true');
      toast.success(t('common.success'));
    }
    finish();
  };

  const steps = [
    // Step 0: Welcome
    {
      icon: Heart,
      title: `${t('onboarding.title').replace('!', '')}, ${userName}!`,
      description: t('onboarding.subtitle'),
      content: (
        <div className="flex gap-3 mt-6">
          <Button onClick={() => setStep(1)} className="flex-1">
            {t('onboarding.next')} <ArrowRight className="ml-2 h-4 w-4" />
          </Button>
          <Button variant="ghost" onClick={finish}>
            {t('onboarding.skipAll')}
          </Button>
        </div>
      ),
    },
    // Step 1: Add photo
    {
      icon: Camera,
      title: t('onboarding.step2'),
      description: t('onboarding.photoPrompt'),
      content: (
        <div className="space-y-4 mt-4">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handlePhotoUpload}
          />
          <div
            onClick={() => fileInputRef.current?.click()}
            className="border-2 border-dashed border-violet-300 dark:border-violet-700 rounded-xl p-8 text-center cursor-pointer hover:bg-violet-50 dark:hover:bg-violet-900/20 transition-colors"
          >
            {photoUploaded ? (
              <div className="flex flex-col items-center gap-2">
                <Check className="h-8 w-8 text-green-500" />
                <p className="text-sm text-green-600 dark:text-green-400 font-medium">{t('common.success')}</p>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-2">
                <Camera className="h-8 w-8 text-violet-400" />
                <p className="text-sm text-muted-foreground">{t('onboarding.photoPrompt')}</p>
              </div>
            )}
          </div>
          <div className="flex gap-3">
            <Button onClick={() => setStep(2)} variant="ghost" className="flex-1">
              {t('onboarding.skip')}
            </Button>
          </div>
        </div>
      ),
    },
    // Step 2: Add medication
    {
      icon: Pill,
      title: t('onboarding.step3'),
      description: t('onboarding.medName'),
      content: (
        <div className="space-y-4 mt-4">
          <div>
            <Label htmlFor="med-name">{t('onboarding.medName')}</Label>
            <Input id="med-name" placeholder="e.g. Aspirin" value={medName} onChange={(e) => setMedName(e.target.value)} />
          </div>
          <div>
            <Label htmlFor="med-dosage">{t('onboarding.medDosage')}</Label>
            <Input id="med-dosage" placeholder="e.g. 100mg" value={medDosage} onChange={(e) => setMedDosage(e.target.value)} />
          </div>
          <div className="flex gap-3">
            <Button onClick={saveMedication} disabled={saving} className="flex-1">
              {saving ? t('common.loading') : t('common.add')} <Check className="ml-2 h-4 w-4" />
            </Button>
            <Button variant="ghost" onClick={() => setStep(3)}>{t('onboarding.skip')}</Button>
          </div>
        </div>
      ),
    },
    // Step 3: Add family member
    {
      icon: Users,
      title: t('onboarding.step4'),
      description: t('onboarding.familyName'),
      content: (
        <div className="space-y-4 mt-4">
          <div>
            <Label htmlFor="fam-name">{t('onboarding.familyName')}</Label>
            <Input id="fam-name" placeholder="e.g. Sarah" value={familyName} onChange={(e) => setFamilyName(e.target.value)} />
          </div>
          <div>
            <Label htmlFor="fam-rel">{t('onboarding.familyRelation')}</Label>
            <Input id="fam-rel" placeholder="e.g. Daughter" value={familyRelationship} onChange={(e) => setFamilyRelationship(e.target.value)} />
          </div>
          <div className="flex gap-3">
            <Button onClick={saveFamilyMember} disabled={saving} className="flex-1">
              {saving ? t('common.loading') : t('common.add')} <Check className="ml-2 h-4 w-4" />
            </Button>
            <Button variant="ghost" onClick={() => setStep(4)}>{t('onboarding.skip')}</Button>
          </div>
        </div>
      ),
    },
    // Step 4: Install PWA
    {
      icon: Download,
      title: t('onboarding.installTitle'),
      description: t('onboarding.installDesc'),
      content: (
        <div className="space-y-4 mt-4">
          <div className="bg-violet-50 dark:bg-violet-900/20 rounded-xl p-6 text-center">
            <Download className="h-12 w-12 text-violet-500 mx-auto mb-3" />
            <p className="text-sm text-muted-foreground mb-4">{t('onboarding.installDesc')}</p>
            <Button onClick={handleInstall} className="w-full">
              {deferredPrompt ? t('onboarding.installButton') : t('onboarding.finish')}
            </Button>
          </div>
          <Button variant="ghost" onClick={finish} className="w-full">
            {t('onboarding.skip')}
          </Button>
        </div>
      ),
    },
  ];

  const current = steps[step];
  const Icon = current.icon;

  return (
    <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4">
      <Card className="w-full max-w-md animate-scale-in">
        <CardHeader className="relative">
          <Button
            variant="ghost"
            size="icon"
            className="absolute right-4 top-4"
            onClick={finish}
          >
            <X className="h-4 w-4" />
          </Button>
          <div className="flex items-center gap-3 mb-2">
            <div className="h-10 w-10 rounded-xl bg-violet-100 dark:bg-violet-900/30 flex items-center justify-center">
              <Icon className="h-5 w-5 text-violet-600 dark:text-violet-400" />
            </div>
            <div className="flex gap-1">
              {Array.from({ length: totalSteps }).map((_, i) => (
                <div
                  key={i}
                  className={`h-1.5 w-8 rounded-full transition-colors ${i <= step ? "bg-violet-600" : "bg-gray-200 dark:bg-gray-700"}`}
                />
              ))}
            </div>
          </div>
          <CardTitle>{current.title}</CardTitle>
          <CardDescription>{current.description}</CardDescription>
        </CardHeader>
        <CardContent>{current.content}</CardContent>
      </Card>
    </div>
  );
};
