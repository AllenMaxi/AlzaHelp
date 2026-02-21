import React from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/context/AuthContext";
import { Button } from "@/components/ui/button";
import {
  Heart,
  LogIn,
  Users,
  Calendar,
  MessageCircle,
  Shield
} from "lucide-react";

export const LoginPage = () => {
  const { login, register, isAuthenticated } = useAuth();
  const navigate = useNavigate();
  const [isLogin, setIsLogin] = React.useState(true);
  const [isLoading, setIsLoading] = React.useState(false);
  const [error, setError] = React.useState("");
  const [successMessage, setSuccessMessage] = React.useState("");
  const [formData, setFormData] = React.useState({
    email: "",
    password: "",
    name: "",
    role: "patient"
  });

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError("");
    setSuccessMessage("");

    try {
      if (isLogin) {
        await login(formData.email, formData.password);
        navigate("/dashboard");
      } else {
        const registration = await register(formData.email, formData.password, formData.name, formData.role);
        if (registration?.requires_approval) {
          setSuccessMessage(
            registration.message || "Registration submitted. Your account is pending admin approval."
          );
          setIsLogin(true);
          setFormData({
            email: formData.email,
            password: "",
            name: "",
            role: "patient"
          });
          return;
        }
        await login(formData.email, formData.password);
        navigate("/dashboard");
      }
    } catch (err) {
      setError(err.message || "An error occurred. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  // Redirect if already authenticated
  React.useEffect(() => {
    if (isAuthenticated) {
      navigate("/dashboard");
    }
  }, [isAuthenticated, navigate]);

  const features = [
    {
      icon: Users,
      title: "Family Directory",
      description: "Keep all your loved ones' information in one safe place"
    },
    {
      icon: Calendar,
      title: "Memory Timeline",
      description: "Store and relive your precious memories anytime"
    },
    {
      icon: MessageCircle,
      title: "AI Assistant",
      description: "Ask questions about your family and memories"
    },
    {
      icon: Shield,
      title: "Safe & Private",
      description: "Your memories are protected and only visible to you"
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-hero">
      {/* Header */}
      <header className="py-6 px-4 sm:px-6 lg:px-8">
        <div className="container mx-auto flex items-center justify-center sm:justify-start">
          <div className="flex items-center gap-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-primary shadow-soft">
              <Heart
                className="h-7 w-7 text-primary-foreground"
                fill="currentColor"
              />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-foreground font-display">
                MemoryKeeper
              </h1>
              <p className="text-sm text-muted-foreground">
                Your memories, always close
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 sm:px-6 lg:px-8 py-12 sm:py-20">
        <div className="grid gap-12 lg:grid-cols-2 lg:gap-20 items-center">
          {/* Left - Auth Form */}
          <div className="max-w-md mx-auto lg:mx-0 w-full animate-fade-in">
            <div className="bg-card border-2 border-border rounded-3xl p-8 shadow-card">
              <div className="text-center mb-8">
                <h2 className="text-2xl font-bold text-foreground mb-2">
                  {isLogin ? "Welcome Back" : "Create Account"}
                </h2>
                <p className="text-muted-foreground">
                  {isLogin
                    ? "Enter your details to access your memories"
                    : "Start preserving your memories today"}
                </p>
              </div>

              {error && (
                <div className="mb-6 p-4 rounded-xl bg-destructive/10 text-destructive text-sm font-medium animate-shake">
                  {error}
                </div>
              )}

              {successMessage && (
                <div className="mb-6 p-4 rounded-xl bg-primary/10 text-primary text-sm font-medium">
                  {successMessage}
                </div>
              )}

              <form onSubmit={handleSubmit} className="space-y-4">
                {!isLogin && (
                  <div>
                    <label className="block text-sm font-medium text-foreground mb-1.5 ml-1">
                      Full Name
                    </label>
                    <input
                      type="text"
                      className="w-full px-4 py-3 rounded-xl border-2 border-input bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all"
                      placeholder="e.g. John Doe"
                      value={formData.name}
                      onChange={(e) =>
                        setFormData({ ...formData, name: e.target.value })
                      }
                      required={!isLogin}
                    />
                  </div>
                )}

                {!isLogin && (
                  <div>
                    <label className="block text-sm font-medium text-foreground mb-1.5 ml-1">
                      Account Role
                    </label>
                    <select
                      className="w-full px-4 py-3 rounded-xl border-2 border-input bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all"
                      value={formData.role}
                      onChange={(e) =>
                        setFormData({ ...formData, role: e.target.value })
                      }
                    >
                      <option value="patient">Patient</option>
                      <option value="caregiver">Caregiver</option>
                      <option value="clinician">Clinician</option>
                      <option value="admin">Admin (Allowlisted)</option>
                    </select>
                  </div>
                )}

                <div>
                  <label className="block text-sm font-medium text-foreground mb-1.5 ml-1">
                    Email Address
                  </label>
                  <input
                    type="email"
                    className="w-full px-4 py-3 rounded-xl border-2 border-input bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all"
                    placeholder="john@example.com"
                    value={formData.email}
                    onChange={(e) =>
                      setFormData({ ...formData, email: e.target.value })
                    }
                    required
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-foreground mb-1.5 ml-1">
                    Password
                  </label>
                  <input
                    type="password"
                    className="w-full px-4 py-3 rounded-xl border-2 border-input bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all"
                    placeholder="••••••••"
                    value={formData.password}
                    onChange={(e) =>
                      setFormData({ ...formData, password: e.target.value })
                    }
                    required
                    minLength={6}
                  />
                </div>

                <Button
                  type="submit"
                  size="xl"
                  className="w-full mt-6"
                  disabled={isLoading}
                >
                  {isLoading ? (
                    <span className="flex items-center gap-2">
                      Processing...
                    </span>
                  ) : (
                    <span className="flex items-center gap-2">
                      <LogIn className="h-5 w-5" />
                      {isLogin ? "Sign In" : "Create Account"}
                    </span>
                  )}
                </Button>
              </form>

              <div className="mt-6 text-center">
                <button
                  onClick={() => {
                    setIsLogin(!isLogin);
                    setError("");
                    setSuccessMessage("");
                    setFormData({ email: "", password: "", name: "", role: "patient" });
                  }}
                  className="text-sm font-medium text-primary hover:text-primary/80 transition-colors"
                >
                  {isLogin
                    ? "Don't have an account? Sign up"
                    : "Already have an account? Sign in"}
                </button>
              </div>
            </div>
          </div>

          {/* Right - Features */}
          <div className="hidden lg:grid gap-4 sm:grid-cols-2">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <div
                  key={feature.title}
                  className="p-6 rounded-2xl bg-card border-2 border-border shadow-card animate-scale-in"
                  style={{ animationDelay: `${index * 0.1}s` }}
                >
                  <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-primary/10 mb-4">
                    <Icon className="h-7 w-7 text-primary" />
                  </div>
                  <h3 className="text-xl font-bold text-foreground mb-2">
                    {feature.title}
                  </h3>
                  <p className="text-base text-muted-foreground">
                    {feature.description}
                  </p>
                </div>
              );
            })}
          </div>
        </div>

        {/* Hero Image */}
        <div className="mt-16 sm:mt-20 relative rounded-3xl overflow-hidden shadow-elevated max-w-4xl mx-auto">
          <img
            src="https://images.unsplash.com/photo-1600779438084-a87b966aab99?crop=entropy&cs=srgb&fm=jpg&ixid=M3w4NTYxODh8MHwxfHNlYXJjaHwxfHxncmFuZHBhcmVudHMlMjBncmFuZGNoaWxkcmVufGVufDB8fHx8MTc2OTM3NzY3N3ww&ixlib=rb-4.1.0&q=85"
            alt="Family memories"
            className="w-full h-[300px] sm:h-[400px] object-cover"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-background/80 via-transparent to-transparent" />
          <div className="absolute bottom-6 left-6 right-6 text-center">
            <p className="text-xl font-semibold text-foreground">
              Your precious memories deserve to be remembered
            </p>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border bg-muted/30 py-8 mt-auto">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <div className="flex items-center justify-center gap-2 mb-3">
            <Heart className="h-5 w-5 text-primary" fill="currentColor" />
            <span className="font-display text-xl font-bold text-foreground">
              MemoryKeeper
            </span>
          </div>
          <p className="text-base text-muted-foreground">
            Made with love to help you remember what matters most.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default LoginPage;
