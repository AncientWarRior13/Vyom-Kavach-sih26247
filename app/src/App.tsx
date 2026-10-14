import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ToastProvider } from './ui/Toast';
import { AppShell } from './ui/AppShell';
import { ErrorBoundary } from './ui/ErrorBoundary';
import { StyleguidePage } from './pages/Styleguide';
import { TrainerPage } from './pages/Trainer';
import { AarPage } from './pages/Aar';
import { InstructorPage } from './pages/Instructor';
import { TeamPage } from './pages/Team';
import { VrPage } from './pages/Vr';
import { SwarmPage } from './pages/Swarm';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

export const App: React.FC = () => {
  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <ToastProvider>
          <BrowserRouter>
            <AppShell>
              <Routes>
                <Route path="/" element={<Navigate to="/trainer" replace />} />
                <Route path="/trainer" element={<TrainerPage />} />
                <Route path="/aar" element={<AarPage />} />
                <Route path="/instructor" element={<InstructorPage />} />
                <Route path="/team" element={<TeamPage />} />
                <Route path="/vr" element={<VrPage />} />
                <Route path="/swarm" element={<SwarmPage />} />
                <Route path="/styleguide" element={<StyleguidePage />} />
                <Route path="*" element={<Navigate to="/trainer" replace />} />
              </Routes>
            </AppShell>
          </BrowserRouter>
        </ToastProvider>
      </QueryClientProvider>
    </ErrorBoundary>
  );
};

export default App;
