const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const rootDir = path.resolve(__dirname, '..');
const distDir = path.join(rootDir, 'dist');
const appDist = path.join(rootDir, 'app', 'dist');
const landingDist = path.join(rootDir, 'landing', 'dist');

console.log('=== Assembling Unified Distribution for Deployment ===');

// Clean dist directory
if (fs.existsSync(distDir)) {
  fs.rmSync(distDir, { recursive: true, force: true });
}
fs.mkdirSync(distDir, { recursive: true });

// 1. Copy landing/dist to root of dist (Landing page at /)
if (fs.existsSync(landingDist)) {
  console.log('Copying landing page assets...');
  fs.copyFileSync(path.join(landingDist, 'index.html'), path.join(distDir, 'index.html'));
  
  const landingStaticDir = path.join(distDir, 'landing-static');
  fs.mkdirSync(landingStaticDir, { recursive: true });
  fs.cpSync(landingDist, landingStaticDir, { recursive: true });
} else {
  console.warn('Warning: landing/dist does not exist yet.');
}

// 2. Copy app/dist to dist (Tactical App)
if (fs.existsSync(appDist)) {
  console.log('Copying tactical app assets and worker scripts...');
  const appEntries = fs.readdirSync(appDist);
  for (const entry of appEntries) {
    const srcPath = path.join(appDist, entry);
    const destPath = path.join(distDir, entry);
    if (entry === 'index.html') {
      fs.copyFileSync(srcPath, path.join(distDir, 'app.html'));
    } else {
      fs.cpSync(srcPath, destPath, { recursive: true });
    }
  }
  
  // Create folders for clean static routes
  const appRoutes = ['trainer', 'aar', 'vr', 'instructor', 'team', 'swarm', 'styleguide'];
  for (const route of appRoutes) {
    const routeDir = path.join(distDir, route);
    fs.mkdirSync(routeDir, { recursive: true });
    fs.copyFileSync(path.join(appDist, 'index.html'), path.join(routeDir, 'index.html'));

    // Ensure worker files are also accessible relative to route directories
    for (const wFile of ['maplibre-gl-worker.mjs', 'maplibre-gl-shared.mjs']) {
      const wSrc = path.join(appDist, wFile);
      if (fs.existsSync(wSrc)) {
        fs.copyFileSync(wSrc, path.join(routeDir, wFile));
      }
    }
  }

  // Ensure worker files are also accessible in assets/ directory where Vite modules run
  const assetsDir = path.join(distDir, 'assets');
  if (fs.existsSync(assetsDir)) {
    for (const wFile of ['maplibre-gl-worker.mjs', 'maplibre-gl-shared.mjs']) {
      const wSrc = path.join(appDist, wFile);
      if (fs.existsSync(wSrc)) {
        fs.copyFileSync(wSrc, path.join(assetsDir, wFile));
      }
    }
  }
} else {
  console.warn('Warning: app/dist does not exist yet.');
}

// 3. Ensure all favicon formats (SVG, multi-resolution binary ICO, PNGs) are available across all routes
const publicDir = path.join(rootDir, 'app', 'public');
const iconFiles = ['favicon.svg', 'favicon.ico', 'favicon-32x32.png', 'favicon-16x16.png', 'apple-touch-icon.png'];

console.log('Copying all favicon formats to root and route directories...');
for (const iconFile of iconFiles) {
  const src = path.join(publicDir, iconFile);
  if (fs.existsSync(src)) {
    fs.copyFileSync(src, path.join(distDir, iconFile));

    const landingStaticDir = path.join(distDir, 'landing-static');
    if (fs.existsSync(landingStaticDir)) {
      fs.copyFileSync(src, path.join(landingStaticDir, iconFile));
    }

    const appRoutes = ['trainer', 'aar', 'vr', 'instructor', 'team', 'swarm', 'styleguide'];
    for (const route of appRoutes) {
      const routeDir = path.join(distDir, route);
      if (fs.existsSync(routeDir)) {
        fs.copyFileSync(src, path.join(routeDir, iconFile));
      }
    }
  }
}

console.log('=== Unified distribution assembled in dist/ ===');
