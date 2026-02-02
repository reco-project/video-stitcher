#!/usr/bin/env node
/**
 * Build Lens Profiles SQLite Database
 *
 * This script converts JSON lens profile files from lens_profiles_src/
 * into a single SQLite database file for efficient runtime access.
 *
 * Usage:
 *   node scripts/build-lens-profiles-sqlite.cjs
 *   node scripts/build-lens-profiles-sqlite.cjs --output path/to/output.sqlite
 *
 * The script is deterministic: same input files produce the same output.
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Default paths
const SRC_DIR = path.join(__dirname, '..', 'backend', 'data', 'lens_profiles');
const DEFAULT_OUTPUT = path.join(__dirname, '..', 'electron', 'resources', 'lens_profiles.sqlite');

// Parse command line arguments
const args = process.argv.slice(2);
let outputPath = DEFAULT_OUTPUT;
let srcDir = SRC_DIR;

for (let i = 0; i < args.length; i++) {
    if (args[i] === '--output' && args[i + 1]) {
        outputPath = args[i + 1];
        i++;
    } else if (args[i] === '--src' && args[i + 1]) {
        srcDir = args[i + 1];
        i++;
    } else if (args[i] === '--help') {
        console.log(`
Build Lens Profiles SQLite Database

Usage:
  node scripts/build-lens-profiles-sqlite.cjs [options]

Options:
  --src <path>      Source directory containing JSON profile files (default: backend/data/lens_profiles)
  --output <path>   Output SQLite file path (default: electron/resources/lens_profiles.sqlite)
  --help            Show this help message
`);
        process.exit(0);
    }
}

/**
 * Recursively find all JSON files in a directory
 */
function findJsonFiles(dir, files = []) {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        if (entry.isDirectory()) {
            findJsonFiles(fullPath, files);
        } else if (entry.isFile() && entry.name.endsWith('.json') && entry.name !== 'LICENSE') {
            files.push(fullPath);
        }
    }
    return files;
}

/**
 * Parse a lens profile JSON file and extract metadata
 */
function parseProfile(filePath) {
    try {
        const content = fs.readFileSync(filePath, 'utf8');
        const profile = JSON.parse(content);

        // Validate required fields
        if (!profile.id || !profile.camera_brand || !profile.camera_model) {
            console.warn(`  Warning: Skipping ${filePath} - missing required fields`);
            return null;
        }

        // Extract resolution
        const w = profile.resolution?.width || null;
        const h = profile.resolution?.height || null;

        // Extract metadata fields
        const metadata = profile.metadata || {};
        const official = metadata.official === true ? 1 : 0;
        const source = metadata.source || null;
        const sourceFile = metadata.source_file || null;
        const notes = metadata.notes || null;

        return {
            id: profile.id,
            camera_brand: profile.camera_brand,
            camera_model: profile.camera_model,
            lens_model: profile.lens_model || null,
            w,
            h,
            distortion_model: profile.distortion_model || null,
            official,
            source,
            source_file: sourceFile,
            notes,
            json: content.trim(),
        };
    } catch (error) {
        console.warn(`  Warning: Failed to parse ${filePath}: ${error.message}`);
        return null;
    }
}

/**
 * Escape a string for SQLite (single quotes become two single quotes)
 */
function escapeSql(str) {
    if (str === null || str === undefined) return 'NULL';
    return "'" + String(str).replace(/'/g, "''") + "'";
}

/**
 * Build the SQLite database using sql.js or sqlite3 CLI
 */
async function buildDatabase() {
    console.log('='.repeat(60));
    console.log('Building Lens Profiles SQLite Database');
    console.log('='.repeat(60));
    console.log(`Source directory: ${srcDir}`);
    console.log(`Output file: ${outputPath}`);
    console.log();

    // Check source directory exists
    if (!fs.existsSync(srcDir)) {
        console.error(`Error: Source directory not found: ${srcDir}`);
        process.exit(1);
    }

    // Find all JSON files
    console.log('Finding JSON profile files...');
    const jsonFiles = findJsonFiles(srcDir);
    console.log(`Found ${jsonFiles.length} JSON files`);
    console.log();

    // Sort files for deterministic output
    jsonFiles.sort();

    // Parse all profiles
    console.log('Parsing profile files...');
    const profiles = [];
    let skipped = 0;
    for (const file of jsonFiles) {
        const profile = parseProfile(file);
        if (profile) {
            profiles.push(profile);
        } else {
            skipped++;
        }
    }
    console.log(`Parsed ${profiles.length} profiles (${skipped} skipped)`);
    console.log();

    // Sort profiles by ID for deterministic output
    profiles.sort((a, b) => a.id.localeCompare(b.id));

    // Ensure output directory exists
    const outputDir = path.dirname(outputPath);
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
    }

    // Remove existing database
    if (fs.existsSync(outputPath)) {
        fs.unlinkSync(outputPath);
    }

    // Build SQL statements
    console.log('Generating SQL...');
    const sqlStatements = [];

    // Create table
    sqlStatements.push(`
CREATE TABLE IF NOT EXISTS profiles (
    id TEXT PRIMARY KEY,
    camera_brand TEXT NOT NULL,
    camera_model TEXT NOT NULL,
    lens_model TEXT,
    w INTEGER,
    h INTEGER,
    distortion_model TEXT,
    official INTEGER DEFAULT 0,
    source TEXT,
    source_file TEXT,
    notes TEXT,
    json TEXT NOT NULL
);
`);

    // Create indexes
    sqlStatements.push(
        'CREATE INDEX IF NOT EXISTS idx_brand_model_lens ON profiles (camera_brand, camera_model, lens_model);'
    );
    sqlStatements.push('CREATE INDEX IF NOT EXISTS idx_resolution ON profiles (w, h);');
    sqlStatements.push('CREATE INDEX IF NOT EXISTS idx_official ON profiles (official);');

    // Create FTS5 table for full-text search (optional but useful)
    sqlStatements.push(`
CREATE VIRTUAL TABLE IF NOT EXISTS profiles_fts USING fts5(
    id,
    camera_brand,
    camera_model,
    lens_model,
    notes,
    content='profiles',
    content_rowid='rowid'
);
`);

    // Insert profiles
    console.log('Generating INSERT statements...');
    for (const profile of profiles) {
        const sql = `INSERT INTO profiles (id, camera_brand, camera_model, lens_model, w, h, distortion_model, official, source, source_file, notes, json) VALUES (
    ${escapeSql(profile.id)},
    ${escapeSql(profile.camera_brand)},
    ${escapeSql(profile.camera_model)},
    ${escapeSql(profile.lens_model)},
    ${profile.w === null ? 'NULL' : profile.w},
    ${profile.h === null ? 'NULL' : profile.h},
    ${escapeSql(profile.distortion_model)},
    ${profile.official},
    ${escapeSql(profile.source)},
    ${escapeSql(profile.source_file)},
    ${escapeSql(profile.notes)},
    ${escapeSql(profile.json)}
);`;
        sqlStatements.push(sql);
    }

    // Populate FTS table
    sqlStatements.push(`
INSERT INTO profiles_fts (rowid, id, camera_brand, camera_model, lens_model, notes)
SELECT rowid, id, camera_brand, camera_model, lens_model, notes FROM profiles;
`);

    // Create triggers for FTS sync (for future writes)
    sqlStatements.push(`
CREATE TRIGGER profiles_ai AFTER INSERT ON profiles BEGIN
    INSERT INTO profiles_fts (rowid, id, camera_brand, camera_model, lens_model, notes)
    VALUES (NEW.rowid, NEW.id, NEW.camera_brand, NEW.camera_model, NEW.lens_model, NEW.notes);
END;
`);

    sqlStatements.push(`
CREATE TRIGGER profiles_ad AFTER DELETE ON profiles BEGIN
    INSERT INTO profiles_fts (profiles_fts, rowid, id, camera_brand, camera_model, lens_model, notes)
    VALUES ('delete', OLD.rowid, OLD.id, OLD.camera_brand, OLD.camera_model, OLD.lens_model, OLD.notes);
END;
`);

    sqlStatements.push(`
CREATE TRIGGER profiles_au AFTER UPDATE ON profiles BEGIN
    INSERT INTO profiles_fts (profiles_fts, rowid, id, camera_brand, camera_model, lens_model, notes)
    VALUES ('delete', OLD.rowid, OLD.id, OLD.camera_brand, OLD.camera_model, OLD.lens_model, OLD.notes);
    INSERT INTO profiles_fts (rowid, id, camera_brand, camera_model, lens_model, notes)
    VALUES (NEW.rowid, NEW.id, NEW.camera_brand, NEW.camera_model, NEW.lens_model, NEW.notes);
END;
`);

    // Write SQL to temp file
    const tempSqlFile = outputPath + '.sql';
    const fullSql = sqlStatements.join('\n');
    fs.writeFileSync(tempSqlFile, fullSql, 'utf8');

    // Execute SQL using sqlite3 CLI
    console.log('Executing SQL with sqlite3...');
    try {
        execSync(`sqlite3 "${outputPath}" < "${tempSqlFile}"`, {
            stdio: 'inherit',
        });
    } catch (error) {
        // If sqlite3 CLI is not available, provide instructions
        console.error('\nError: sqlite3 CLI not found or failed.');
        console.error('Please install sqlite3:');
        console.error('  - Ubuntu/Debian: sudo apt-get install sqlite3');
        console.error('  - macOS: brew install sqlite3');
        console.error('  - Windows: choco install sqlite or download from https://sqlite.org/download.html');
        console.error('\nAlternatively, you can use the generated SQL file manually:');
        console.error(`  ${tempSqlFile}`);
        process.exit(1);
    }

    // Clean up temp SQL file
    fs.unlinkSync(tempSqlFile);

    // Verify database
    console.log('\nVerifying database...');
    try {
        const countOutput = execSync(`sqlite3 "${outputPath}" "SELECT COUNT(*) FROM profiles;"`, {
            encoding: 'utf8',
        }).trim();
        const count = parseInt(countOutput, 10);
        console.log(`  Profiles in database: ${count}`);

        if (count !== profiles.length) {
            console.error(`  ERROR: Expected ${profiles.length} profiles, got ${count}`);
            process.exit(1);
        }

        // Check indexes
        const indexOutput = execSync(
            `sqlite3 "${outputPath}" "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%';"`,
            {
                encoding: 'utf8',
            }
        ).trim();
        const indexes = indexOutput.split('\n').filter((x) => x);
        console.log(`  Indexes created: ${indexes.length}`);

        // Check FTS
        const ftsOutput = execSync(
            `sqlite3 "${outputPath}" "SELECT COUNT(*) FROM profiles_fts WHERE profiles_fts MATCH 'gopro';"`,
            {
                encoding: 'utf8',
            }
        ).trim();
        console.log(`  FTS test (search 'gopro'): ${ftsOutput} results`);
    } catch (error) {
        console.error(`  Verification failed: ${error.message}`);
        process.exit(1);
    }

    // Show file size
    const stats = fs.statSync(outputPath);
    const sizeMB = (stats.size / (1024 * 1024)).toFixed(2);
    console.log(`  Database size: ${sizeMB} MB`);

    console.log('\n' + '='.repeat(60));
    console.log('SUCCESS: Lens profiles database built!');
    console.log(`Output: ${outputPath}`);
    console.log('='.repeat(60));
}

// Run
buildDatabase().catch((error) => {
    console.error('Build failed:', error);
    process.exit(1);
});
