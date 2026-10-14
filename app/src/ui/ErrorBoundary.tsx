import React, { Component, ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RotateCcw } from 'lucide-react';
import { Button } from './Button';

interface Props {
  children: ReactNode;
  fallbackTitle?: string;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Tactical UI Uncaught Error:', error, errorInfo);
  }

  private handleReset = () => {
    this.setState({ hasError: false, error: null });
    window.location.reload();
  };

  public render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-[#020508] text-[#f0f8ff] flex items-center justify-center p-6 select-none font-mono">
          <div className="max-w-md w-full bg-[#08121e] border border-[#ff3b3b]/60 rounded-xs p-6 shadow-[0_0_30px_rgba(255,59,59,0.2)] space-y-4">
            <div className="flex items-center gap-3 text-[#ff3b3b]">
              <AlertTriangle className="w-6 h-6 animate-pulse" />
              <h2 className="font-['Chakra_Petch'] font-bold text-lg tracking-wider uppercase">
                {this.props.fallbackTitle || 'TACTICAL SUBSYSTEM RECOVERY'}
              </h2>
            </div>
            
            <p className="text-xs text-[#8aaabf] leading-relaxed">
              A runtime anomaly interrupted radar display execution. Diagnostics have been logged to the tactical console.
            </p>

            {this.state.error && (
              <div className="p-3 bg-[#020508] border border-[#1e3c64] rounded-xs text-[11px] text-[#ff3b3b] overflow-x-auto font-mono">
                {this.state.error.message}
              </div>
            )}

            <div className="pt-2 flex items-center justify-end">
              <Button
                variant="danger"
                size="sm"
                onClick={this.handleReset}
                icon={<RotateCcw className="w-3.5 h-3.5" />}
              >
                REBOOT CONSOLE
              </Button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
