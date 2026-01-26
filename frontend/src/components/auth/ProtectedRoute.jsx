import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '@/context/AuthContext';
import { Loader2, Heart } from 'lucide-react';

export const ProtectedRoute = ({ children }) => {
  const { isAuthenticated, loading, user } = useAuth();
  const location = useLocation();

  // Skip auth check if user was passed from AuthCallback
  const hasUserFromCallback = location.state?.user;

  if (loading && !hasUserFromCallback) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-hero">
        <div className="text-center">
          <div className="flex items-center justify-center mb-6">
            <div className="h-16 w-16 rounded-2xl bg-primary flex items-center justify-center shadow-card animate-pulse-gentle">
              <Heart className="h-8 w-8 text-primary-foreground" fill="currentColor" />
            </div>
          </div>
          <div className="flex items-center justify-center gap-3">
            <Loader2 className="h-6 w-6 animate-spin text-primary" />
            <span className="text-xl text-muted-foreground">Loading your memories...</span>
          </div>
        </div>
      </div>
    );
  }

  if (!isAuthenticated && !hasUserFromCallback) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return children;
};

export default ProtectedRoute;
