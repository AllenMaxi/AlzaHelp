import React, { useEffect, useRef } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '@/context/AuthContext';
import { Loader2, Heart } from 'lucide-react';

export const AuthCallback = () => {
  const { processSessionId } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const hasProcessed = useRef(false);

  useEffect(() => {
    // Prevent double processing in StrictMode
    if (hasProcessed.current) return;
    hasProcessed.current = true;

    const processAuth = async () => {
      // Extract session_id from URL hash
      const hash = location.hash;
      const sessionIdMatch = hash.match(/session_id=([^&]+)/);
      
      if (sessionIdMatch) {
        const sessionId = sessionIdMatch[1];
        try {
          const user = await processSessionId(sessionId);
          // Navigate to dashboard with user data
          navigate('/dashboard', { state: { user }, replace: true });
        } catch (error) {
          console.error('Auth callback error:', error);
          navigate('/login', { replace: true });
        }
      } else {
        navigate('/login', { replace: true });
      }
    };

    processAuth();
  }, []);

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
          <span className="text-xl text-muted-foreground">Signing you in...</span>
        </div>
      </div>
    </div>
  );
};

export default AuthCallback;
