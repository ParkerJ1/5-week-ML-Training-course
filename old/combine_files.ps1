# --- Configuration ---
$OutputFilename = "Training_Program.pdf"
$HeaderFile = "header.tex"
$PDFEngine = "xelatex"

# Define the file order
$Files = @(
    "Week1_Overview.md"
    "Week1_Day1.md"
    "Week1_Day2.md"
    "Week1_Day3.md"
    "Week1_Day4.md"
    "Week1_Day5.md"
    "Week2_Overview.md"
    "Week2_Day6.md"
    "Week2_Day7.md"
    "Week2_Day8.md"
    "Week2_Day9.md"
    "Week2_Day10.md"
)

# --- 1. Concatenate the actual file content ---
$FullContent = foreach ($File in $Files) {
    # Get the content of the actual markdown file
    Get-Content -Path $File -Raw
    
    # If it's not the last file, insert the page break content
    if ($File -ne $Files[-1]) {
        # Get the content of the newpage.md file
        Get-Content -Path "newpage.md" -Raw
    }
}

# --- 2. Pipe the entire consolidated content to Pandoc ---
$FullContent | Out-String -Stream | pandoc `
    --toc `
	--toc-depth=1 `
    --pdf-engine=$PDFEngine `
    -H $HeaderFile `
    --from markdown `
    -o $OutputFilename