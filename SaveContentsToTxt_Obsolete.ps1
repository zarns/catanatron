# Define the root directory
$rootDir = "C:\Users\mason\programming\catanatron"

# Define the paths of the pertinent directories
$pertinentDirs = @(
    "catanatron_core/catanatron",
    "catanatron_core/catanatron/models",
    "catanatron_experimental/catanatron_experimental",
    "catanatron_experimental/catanatron_experimental/cli",
    "catanatron_experimental/catanatron_experimental/machine_learning",
    "catanatron_gym/catanatron_gym",
    "catanatron_server/catanatron_server",
    "tests",
    "ui/src"
)

# Paths to the output files
$directoryStructureFilePath = "directory_structure.txt"
$filesContentFilePath = "files_content.txt"

# Function to print the directory structure
function PrintDirectoryStructure {
    param (
        [string]$dir,
        [string]$indent = ""
    )
    
    $items = Get-ChildItem -Path $dir -Recurse -Force -File

    foreach ($item in $items) {
        $relativePath = $item.FullName.Substring($rootDir.Length + 1).Replace("\", "/")
        Write-Host "Checking: $relativePath"
        $isInPertinentDir = $false
        foreach ($pertinentDir in $pertinentDirs) {
            if ($relativePath.StartsWith($pertinentDir)) {
                Write-Host "Match found in pertinentDirs: $pertinentDir"
                $isInPertinentDir = $true
                break
            }
        }

        if ($isInPertinentDir -and -not $relativePath.Contains("node_modules") -and -not $relativePath.EndsWith(".pyc")) {
            $indentLevel = ($relativePath.Split("/") | Measure-Object).Count - 1
            $indentation = " " * $indentLevel
            Write-Host "File: $relativePath"
            Add-Content -Path $directoryStructureFilePath -Value "$indentation$item"
            SaveFileContents -filePath $relativePath
        } else {
            Write-Host "Not in pertinentDirs or excluded: $relativePath"
        }
    }
}

# Function to save file contents to a single output file
function SaveFileContents {
    param (
        [string]$filePath
    )
    
    $fullPath = Join-Path -Path $rootDir -ChildPath $filePath
    Write-Host "Attempting to read: $fullPath"

    if (Test-Path $fullPath) {
        Write-Host "Reading file: $fullPath"
        Add-Content -Path $filesContentFilePath -Value "`n--- $filePath ---`n"
        Get-Content -Path $fullPath | Add-Content -Path $filesContentFilePath
    } else {
        Write-Host "File not found: $fullPath"
        Add-Content -Path $filesContentFilePath -Value "`n--- $filePath not found ---`n"
    }
}

# Clear the output files if they exist
if (Test-Path $directoryStructureFilePath) {
    Write-Host "Removing existing $directoryStructureFilePath"
    Remove-Item $directoryStructureFilePath
}
if (Test-Path $filesContentFilePath) {
    Write-Host "Removing existing $filesContentFilePath"
    Remove-Item $filesContentFilePath
}

# Ensure the output files are created
Write-Host "Creating $directoryStructureFilePath"
New-Item -ItemType File -Path $directoryStructureFilePath -Force
Write-Host "Creating $filesContentFilePath"
New-Item -ItemType File -Path $filesContentFilePath -Force

# Call the function to print the directory structure and save file contents
Write-Host "Starting directory scan..."
PrintDirectoryStructure -dir $rootDir

Write-Host "Directory structure has been saved to $directoryStructureFilePath"
Write-Host "File contents have been saved to $filesContentFilePath"
