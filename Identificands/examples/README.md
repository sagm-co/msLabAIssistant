# ***Identificands*: Basic Usage**

*Identificands* requires two types of input files: (1) one or more inclusion lists, and (2) a list of raw data files.


### Inclusion Lists
An inclusion list is a `.tsv` file containing seven columns: `inchikey`, `name`, `smiles`, `ionizedSpecies`, `fragmentationProfile`, `tR`, and `amenability`. The data provided in these fields dictates the analysis mode for each row.

* **Targeted Analysis (TA):** Performed when all fields are completed. The `amenability` field must be `1`, and `fragmentationProfile` must contain a semicolon-separated list of *m/z* values.
* **Suspect Screening (SS):** Performed if one or more of the following fields are empty: `ionizedSpecies`, `fragmentationProfile`, `tR`, or `amenability`.
* **Non-targeted Screening (NTS):** Performed if the `smiles` field contains only an `X`. *Note: NTS currently supports only (de)protonated species.*

A single inclusion list can contain mixed rows for different identification approaches, or they can be separated across multiple files. Inclusion list files are located in `inputData/<inclusionListName>.tsv`.


### Raw File List
The raw file list is a `.tsv` file containing two columns:
* `fileToProcess`: The file path to the raw data file.
* `inclusionList`: The associated inclusion list, which defines the identification approaches for that specific raw file.

This raw file list is located in `inputData/<samplesFile>.tsv`.

### Usage Example
Execute the following commands to launch the Jupyter Notebook, then run all cells:

```bash
cd examples
bash samplesAnalysis.sh
```

Identification results are saved in the `resultsData` directory.


### Results

Results are stored in the `resultsData` directory. Inside each sample directory, specific files are generated depending on the identification approach used. The `<identificationApproach>` prefix will be one of the following: `targetedAnalysis` (TA), `suspectScreening` (SS), `nonTargetedScreening` (NTS), or `targetedScreening` (when TA and SS are combined in the same inclusion list). 

The generated files are:

* `<sampleName>_summaryResults.tsv`: Filtered identification results scored by confidence across all approaches, including identificand probabilities and their effect on identification confidence.
* `<sampleName>_identificandsProbabilityWithoutFiltering.tsv`: Calculated identificand probabilities for all analytical signals and NTS candidates prior to filtering.
* `<sampleName>_<identificationApproach>AllDetectedCompounds.tsv`: Data for all detected analytical signals.
* `<sampleName>_<identificationApproach>FragmentsIonsFound.tsv`: Probable fragment ions for a given compound that overlap with the Full Scan (FS) signal.
* `<sampleName>_<identificationApproach>FragmentsIonsFoundAnnotated.tsv`: Annotated fragments associated with a given compound. When combined with `<sampleName>_<identificationApproach>FragmentsIonsFound.tsv`, it provides the complete MS/MS spectra for the compound.
* `<sampleName>_<identificationApproach>FSXICs.tsv` and `<sampleName>_<identificationApproach>XICs4fragmentsIonsFound.tsv`: Extracted Ion Chromatograms (XICs) for both FS and MS/MS vDIA for the detected signals.


