import React from 'react';
import { Panel } from '../ui/Panel';
import { Button } from '../ui/Button';
import { Link } from 'react-router-dom';

export interface PlaceholderPageProps {
  title: string;
  subtitle: string;
  description: string;
  legacyUrl: string;
  icon?: string;
  actionLabel?: string;
}

export const PlaceholderPage: React.FC<PlaceholderPageProps> = ({
  title,
  subtitle,
  description,
  legacyUrl,
  icon = '🎯',
}) => {
  return (
    <div className="p-6 max-w-4xl mx-auto space-y-6">
      <Panel title={title} subtitle={subtitle}>
        <div className="p-8 flex flex-col items-center text-center space-y-4">
          <div className="text-4xl">{icon}</div>
          <h2 className="font-display text-xl font-bold uppercase tracking-wider text-[var(--c-text-max)]">
            {title}
          </h2>
          <p className="font-mono text-xs text-[var(--c-text-mid)] max-w-md">
            {description}
          </p>
          <div className="flex items-center gap-3 pt-4">
            <Link to="/styleguide">
              <Button variant="secondary">View Design System</Button>
            </Link>
            <a href={legacyUrl} target="_blank" rel="noopener noreferrer">
              <Button variant="outline">Launch Legacy Screen ↗</Button>
            </a>
          </div>
        </div>
      </Panel>
    </div>
  );
};
