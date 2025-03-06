# Folder to monitor for Excel file creation
$sourceFolder = "<SOURCE_FOLDER_PATH>"

# Folder where CSV files will be saved
$csvDestinationFolder = "<CSV_DESTINATION_FOLDER_PATH>"

# Folder where the GZIP files will be moved after compression
$gzipDestinationFolder = "<GZIP_DESTINATION_FOLDER_PATH>"

# Path to azcopy folder
$azCopyLocation = "<AZCOPY_FOLDER_PATH>"

# Base URL for Blob Storage
$blobContainerBaseUrl = "<BLOB_CONTAINER_BASE_URL>"

# SAS token for authentication (sensitive data removed)
$sasToken = "<SAS_TOKEN>"

# Change to the directory where azcopy is located
Set-Location -Path $azCopyLocation

# Set up FileSystemWatcher to monitor file creation, changes, and renaming in the source folder
$fsw = New-Object IO.FileSystemWatcher $sourceFolder
$fsw.IncludeSubdirectories = $false  # Do not monitor subdirectories
$fsw.EnableRaisingEvents = $true     # Enable monitoring

# Action when a file is created, changed, or renamed
$action = {
    $newFilePath = $Event.SourceEventArgs.FullPath
    
    # Only process files, not directories
    if (Test-Path $newFilePath -PathType Leaf) {
        
        # Check if the file is not a temporary file 
        if ($newFilePath -match "\.crdownload$|\.tmp$") {
            Write-Host "Skipping temporary file: $newFilePath"
            return
        }

        # Wait for the file to be fully created or renamed 
        Start-Sleep -Seconds 5  # Wait to ensure download or file creation is complete

        try {
            # Get the current date and time 
            $dateTime = Get-Date -Format "yyyyMMddHHmmss"
            # Get the original file name and extension
            $originalFileName = [System.IO.Path]::GetFileName($newFilePath)
            # Construct the new file name with the date prefix
            $newFileName = "$dateTime`_$originalFileName"
            
            # Get the directory of the file
            $directory = [System.IO.Path]::GetDirectoryName($newFilePath)
            
            # Combine the directory path with the new file name
            $newFilePathWithDate = Join-Path $directory $newFileName
            
            # Rename the file
            Rename-Item $newFilePath -NewName $newFilePathWithDate
            Write-Host "File renamed to: $newFilePathWithDate"

            # Initialize Excel COM object
            $excel = New-Object -ComObject Excel.Application
            $excel.Visible = $false  # Keep Excel invisible
            
            # Open the Excel file
            $workbook = $excel.Workbooks.Open($newFilePathWithDate)
            
            # Construct the CSV file path in the CSV destination folder
            $csvFilePath = Join-Path $csvDestinationFolder ([System.IO.Path]::GetFileNameWithoutExtension($newFilePathWithDate) + ".csv")
            
            # Save the workbook as CSV in the CSV destination folder
            $workbook.SaveAs($csvFilePath, 6)  

            # Close the workbook and Excel application
            $workbook.Close()
            $excel.Quit()

            Write-Host "Excel file converted to CSV and saved to: $csvFilePath"

            # Optionally, delete the original Excel file after conversion
            Remove-Item $newFilePathWithDate -Force
            Write-Host "Original Excel file deleted: $newFilePathWithDate"

            # Convert the CSV file to UTF-8 encoding
            $utf8CsvFilePath = [System.IO.Path]::ChangeExtension($csvFilePath, "csv.csv")
            Get-Content $csvFilePath | Out-File -FilePath $utf8CsvFilePath -Encoding utf8
            Write-Host "CSV file converted to UTF-8 and saved to: $utf8CsvFilePath"

            # Compress the UTF-8 CSV file into a .gz file using GZIP
            $gzipFilePath = [System.IO.Path]::ChangeExtension($utf8CsvFilePath, ".gz")
            # Ensure the CSV file exists
            if (Test-Path $utf8CsvFilePath) {
                # Open the CSV file and create the Gzip file
                $csvStream = [System.IO.File]::OpenRead($utf8CsvFilePath)
                $gzipStream = [System.IO.File]::Create($gzipFilePath)
                $compressionStream = New-Object System.IO.Compression.GzipStream($gzipStream, [System.IO.Compression.CompressionMode]::Compress)
                $csvStream.CopyTo($compressionStream)
                # Close streams
                $compressionStream.Close()
                $csvStream.Close()

                Write-Host "CSV file compressed to GZIP: $gzipFilePath"

                # Move the GZIP file to the final destination folder
                $finalGzipFilePath = Join-Path $gzipDestinationFolder ([System.IO.Path]::GetFileName($gzipFilePath))
                Move-Item $gzipFilePath -Destination $finalGzipFilePath -Force
                Write-Host "GZIP file moved to: $finalGzipFilePath"

                # Determine which container to upload to based on the file name
                if ($newFileName -match "BuS") {
                    $blobContainerUrl = "$blobContainerBaseUrl" + "BuS/"
                } elseif ($newFileName -match "kdrs" -or $newFileName -match "CustomerReturnCockpitDetailExport") {
                    $blobContainerUrl = "$blobContainerBaseUrl" + "kdrs/"
                }

                # Full destination URL, including the file name and SAS token
                $destinationUrl = "$blobContainerUrl" + [System.IO.Path]::GetFileName($finalGzipFilePath) + "$sasToken"

                Write-Host "Uploading file to: $destinationUrl"

                # Run azcopy to upload the file to Azure Blob Storage
                $azcopyCommand = .\azcopy copy "$finalGzipFilePath" "$destinationUrl"

                # Check if azcopy command was successful
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "Upload completed. File uploaded to: $destinationUrl"
                } else {
                    Write-Host "AzCopy failed with exit code: $LASTEXITCODE"
                }
            } else {
                Write-Host "UTF-8 CSV file not found: $utf8CsvFilePath"
            }
        } catch {
            Write-Host "Error during Excel to CSV conversion, UTF-8 conversion or compression: $_"
        }
    }
}

# Register events for created, changed, and renamed files
Register-ObjectEvent $fsw "Created" -Action $action
Register-ObjectEvent $fsw "Changed" -Action $action
Register-ObjectEvent $fsw "Renamed" -Action $action

Write-Host "Monitoring folder for Excel file creation, changes, and renaming: $sourceFolder"

# Keep the script running
while ($true) {
    Start-Sleep -Seconds 1  # Keep the script running
}