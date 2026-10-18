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
  console.log('Copying tactical app assets...');
  if (fs.existsSync(path.join(appDist, 'assets'))) {
    fs.cpSync(path.join(appDist, 'assets'), path.join(distDir, 'assets'), { recursive: true });
  }
  fs.copyFileSync(path.join(appDist, 'index.html'), path.join(distDir, 'app.html'));
  
  // Create folders for clean static routes
  const appRoutes = ['trainer', 'aar', 'vr', 'instructor', 'team', 'swarm', 'styleguide'];
  for (const route of appRoutes) {
    const routeDir = path.join(distDir, route);
    fs.mkdirSync(routeDir, { recursive: true });
    fs.copyFileSync(path.join(appDist, 'index.html'), path.join(routeDir, 'index.html'));
  }
} else {
  console.warn('Warning: app/dist does not exist yet.');
}

// 3. Ensure favicon.svg is universally available in root and all static route folders
const faviconSrc = path.join(rootDir, 'app', 'public', 'favicon.svg');
if (fs.existsSync(faviconSrc)) {
  console.log('Copying favicon.svg to root and route directories...');
  fs.copyFileSync(faviconSrc, path.join(distDir, 'favicon.svg'));
  fs.copyFileSync(faviconSrc, path.join(distDir, 'favicon.ico'));

  const landingStaticDir = path.join(distDir, 'landing-static');
  if (fs.existsSync(landingStaticDir)) {
    fs.copyFileSync(faviconSrc, path.join(landingStaticDir, 'favicon.svg'));
  }

  const appRoutes = ['trainer', 'aar', 'vr', 'instructor', 'team', 'swarm', 'styleguide'];
  for (const route of appRoutes) {
    const routeDir = path.join(distDir, route);
    if (fs.existsSync(routeDir)) {
      fs.copyFileSync(faviconSrc, path.join(routeDir, 'favicon.svg'));
    }
  }
}

console.log('=== Unified distribution assembled in dist/ ===');
