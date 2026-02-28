import React from "react";
import { Button } from "@/components/ui/button";
import { AlertTriangle } from "lucide-react";

export class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error, info) {
    console.error("ErrorBoundary caught:", error, info);
    if (window.posthog) {
      window.posthog.capture("client_error", {
        error: error?.message,
        componentStack: info?.componentStack,
      });
    }
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback;
      const isApp = this.props.level === "app";
      return (
        <div className="flex flex-col items-center justify-center p-8 text-center gap-4">
          <AlertTriangle className="h-12 w-12 text-amber-500" />
          <h2 className="text-lg font-semibold">
            {isApp ? "Something went wrong" : "This section encountered an error"}
          </h2>
          <p className="text-muted-foreground text-sm max-w-md">
            {isApp
              ? "Please refresh the page. If the problem persists, try clearing your browser cache."
              : "Try again or switch to a different tab."}
          </p>
          <Button
            onClick={() => this.setState({ hasError: false })}
            variant="outline"
          >
            Try Again
          </Button>
        </div>
      );
    }
    return this.props.children;
  }
}
