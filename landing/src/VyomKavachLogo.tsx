import React from 'react';

export const VyomKavachLogo: React.FC<{ className?: string; size?: number }> = ({
  className = 'w-5 h-5 drop-shadow-[0_0_8px_rgba(0,229,255,0.45)]',
  size = 20,
}) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 64 64"
    width={size}
    height={size}
    className={className}
    style={{ display: 'inline-block', verticalAlign: 'middle' }}
    aria-label="Vyom Kavach Emblem"
  >
    <defs>
      <linearGradient id="vkLandingTrishulaGrad" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" stopColor="#38F8D4" />
        <stop offset="50%" stopColor="#00E5FF" />
        <stop offset="100%" stopColor="#0F243E" />
      </linearGradient>
      <linearGradient id="vkLandingRingGrad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stopColor="#00E5FF" />
        <stop offset="100%" stopColor="#0284C7" />
      </linearGradient>
      <filter id="vkLandingAstraGlow" x="-20%" y="-20%" width="140%" height="140%">
        <feGaussianBlur stdDeviation="1.5" result="blur" />
        <feMerge>
          <feMergeNode in="blur" />
          <feMergeNode in="SourceGraphic" />
        </feMerge>
      </filter>
    </defs>

    <g transform="translate(32, 32)">
      <circle
        cx="0"
        cy="0"
        r="23"
        fill="none"
        stroke="url(#vkLandingRingGrad)"
        strokeWidth="2"
        strokeDasharray="14 6"
        opacity="0.6"
      />
      <circle
        cx="0"
        cy="0"
        r="16"
        fill="none"
        stroke="#00E5FF"
        strokeWidth="1.2"
        opacity="0.3"
      />

      <path
        d="M -7,6 L -22,-6 L -24,-18 L -17,-12 L -7,-3 Z"
        fill="url(#vkLandingTrishulaGrad)"
        stroke="#38F8D4"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />

      <path
        d="M 7,6 L 22,-6 L 24,-18 L 17,-12 L 7,-3 Z"
        fill="url(#vkLandingTrishulaGrad)"
        stroke="#38F8D4"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />

      <path
        d="M 0,-27 L 8,-8 L 4,14 L 0,22 L -4,14 L -8,-8 Z"
        fill="#0A192F"
        stroke="#00E5FF"
        strokeWidth="2.2"
        strokeLinejoin="round"
        filter="url(#vkLandingAstraGlow)"
      />

      <polygon points="0,-16 4,-4 0,4 -4,-4" fill="#38F8D4" filter="url(#vkLandingAstraGlow)" />
      <circle cx="0" cy="-4" r="1.5" fill="#FFFFFF" />

      <path
        d="M -16,14 Q 0,26 16,14"
        fill="none"
        stroke="#00E5FF"
        strokeWidth="2.5"
        strokeLinecap="round"
      />
    </g>
  </svg>
);
