/// <summary> 
/// Implementacion de lector de archivos .Raw empleando C#
/// @brief Lector en consola de archivos .Raw generados de mediciones instrumentales empleando sistema LC-MS-Orbitrap Q Exactive
/// @author Sergio A. Gonzalez-Monico (SAGM) sagonzalezm@unal.edu.co
/// @copyright Grupo de investigacion **Residualidad y destino ambiental de plaguicidas** - Departamento de Quimica
/// @copyright Facultad de Ciencias - Universidad Nacional de Colombia
/// @date 2022-04-10
/// </summary>
///
/// <description>
/// RawFileReader es la mas reciente implementacion de Thermo para lectura de archivos .Raw, en remplazo del lector MSFileReader.
/// Esta implementacion hace uso del conjunto de funciones .NET ensambladas en las librerias de RawFileReader.
/// Las librerias fueron obtenidas del repositorio de Github [thermofisherlsms](https://github.com/thermofisherlsms) por sugerencia de Jim Shofstahl (jim@Shofstahl.thermofisher.com), donde se encuentran dispobibles en bibliotecas [NuGet](https://www.nuget.org). La implementacion se basa en la documentacion de dicho repostorio, sin hacer uso de fuentes de codigo de Thermo Fisher.
///
/// Esta implementacion fue compilada y ejecutada satisfactoriamente en:
///
///   - Windows 11: empleando Microsoft Visual Studio 2022
///   - Linux Ubuntu 20.04 empleando [Mono](https://www.mono-project.com) y [dotnet](https://dotnet.microsoft.com)
///
/// Se hace uso de las bibliotecas  ThermoFisher.CommonCore version 5.0.0.71.
///
namespace LARPRawReader
{
    /// <summary>
    /// Librerias requeridas para lectura de archivos .Raw
    /// <summary>
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.IO;
    using System.Linq;
    using System.Runtime.ExceptionServices;
    using System.Text.RegularExpressions;


    using ThermoFisher.CommonCore.Data.FilterEnums;	
    using ThermoFisher.CommonCore.Data.Business;
    using ThermoFisher.CommonCore.Data;
    using ThermoFisher.CommonCore.MassPrecisionEstimator;
    using ThermoFisher.CommonCore.RawFileReader;
    using ThermoFisher.CommonCore.Data.Interfaces;	


    /// <summary>
    /// Programa principal para acceso a información de enventos asociados a experimentos de adquición.
    /// - Método Tune
    /// - Método instrumental
    /// - Información de la muestra
    /// - Informacion sobre enventos de scan
    ///    -# Metadatos
    ///    -# Espectros MS
    /// </summary>
    internal static class Program
    {
	/// @public
    /// @brief  Access a .Raw file to get an specified information 
	///
	/// @param RawFile  'RawFile' Binary .Raw file to access
	/// @param -g <infoType> type information to get from the binary Raw file. The **infoType** options include:
	///      - tuneInfo -n <ntune>
	///      - instMethodInfo: information about chromatographic and MS methods
	///      - sampleInfo (by default)
	///      - scanInfo -n <nscan>
	///      - scanData -n <nscan>
	///		 - eventsInfo: information about all events into the .Raw file
	///		 - rawAnalysisInfo: infomation about the instrumental analysis (time range, number of scans,  mass range, mode, filters, etc.)
	///		 - experimentsInfo: information about MS experiments
	///      - scanFilterInfo -n <nscan>
	///		 - filtersInfo
	///      - chromatogram: return a chromatogram with the specified settings (-trace, -mzAcc, -massRange)
	/// @param -n <ntune/nscan> Tune/Scan index number to retrieve information (By default 0:tune, 1:scan ).
	/// @param -t <profile/centroid (by default)>
	/// @param -f <true/false> show filters info. False by default
	/// @param -trace <basePeak (by default)/massRange/tic/specMax> kind of trace
	/// @param -mzFilter <filterString> string with filter specifications
	/// @param -mzAcc <double> mass accuracy value in ppm
	/// @param -massRange <lowValue,highValue> mass range for the trace (double values). If highRange<0.0 and lowRange>0.0 the mass range is set with mzAcc around the lowRange value.
	//   [-ti <data, calibration, configuration, identification>]
        private static void Main(string[] args)
        {
 
            try
            {
                string filename = string.Empty;
				
                if (args.Length > 0)
                {
                    filename = args[0];
                }

                if (string.IsNullOrEmpty(filename))
                {
                    Console.WriteLine("No se ha especificado ningún archivo de entrada");
                    return;
                }

                if (!File.Exists(filename))
                {
                    Console.WriteLine(@"No se encuentra el archivo {0}",filename);
                    return;
                }

                var rawFile = RawFileReaderAdapter.FileFactory(filename);

                if (!rawFile.IsOpen || rawFile.IsError)
                {
					
                    Console.WriteLine("Error {0}, no se puede acceder al archivo {1}", rawFile.FileError, filename);
                    return;
                }

                if (rawFile.IsError)
                {
                    Console.WriteLine("Error ({0}) al abrir el archivo {1}", rawFile.FileError, filename);
                    return;
                }

                if (rawFile.InAcquisition)
                {
                    Console.WriteLine("El archivo {0}  no se ha terminado de adquirir ", filename);
                    return;
                }

                rawFile.SelectInstrument(Device.MS, 1);
				int maxScan=rawFile.RunHeader.LastSpectrum;
				int indexST=0;
				string infoType = "sampleInfo";
				string scanType = "centroid";
				string traceType = "basePeak";
				string mzFilter = "";
				double mzAcc=0.0;
				bool isFilterInfo = false;
				var massRange= new double[2]{0,-1.0};


                if (args.Length > 1)
                {
					var rg= new Regex(@"-[aA-zZ][aA-zZ]*");
					string allArgs=String.Join(" ",args,1,args.Length-1);
					MatchCollection opts=rg.Matches(allArgs);

					for (int count = 0; count < opts.Count; count++)
					{
						int argValIndx=count*2+2;
						try
						{
							switch(opts[count].Value)
							{
								case "-g":
									infoType=args[argValIndx];
								break;

								case "-n":
									indexST=int.Parse(args[argValIndx]);
									if(indexST>maxScan) indexST=maxScan;
									if(indexST<0) indexST=0;
								break;

								case "-t":
									scanType=args[argValIndx];
								break;

								case "-f":
									isFilterInfo = bool.Parse(args[argValIndx]);
								break;

								case "-trace":
									traceType = args[argValIndx];
								break;

								case "-mzFilter":
									mzFilter = args[argValIndx];
								break;

								case "-mzAcc":
									mzAcc = double.Parse(args[argValIndx]);
								break;

								case "-massRange":
									var rg2= new Regex(@"-?[\.0-9]+");
									var mR = args[argValIndx];
									MatchCollection lims=rg2.Matches(mR);
									massRange[0]= double.Parse(lims[0].Value);
									massRange[1]= double.Parse(lims[1].Value);						
								break;



							}
						}
						catch
						{
						}
					}

					/// Range mass setting
					if( massRange[1]<=0.0 && massRange[0]>0.0)
					{
						massRange[1]=massRange[0]*(1.0+mzAcc*1E-6);
						massRange[0]=massRange[0]*(1.0-mzAcc*1E-6);
					}

					switch (infoType)
					{
						case "tuneInfo":
							GetTuneData(rawFile, indexST);
						break;

						case "instMethodInfo":
							GetInstrumentalMethods(rawFile);
						break;

						case "elutionInfo":
							GetElutionMethod(rawFile);
						break;

						case "filtersInfo":						
							GetFiltersInfo(rawFile, isFilterInfo);
						break;

						case "scanFilterInfo":
							if(indexST==0){indexST=1;}
							GetFilterInfoForScan(rawFile, indexST);
						break;

						case "experimentsInfo":
							GetExperimentsInfo(rawFile);
						break;

						case "rawAnalysisInfo":
							GetRawAnalysisInfo(rawFile);
						break;

						case "eventsInfo":
							GetEventsInfo(rawFile);
						break;

						case "sampleInfo":
							GetSampleInformation(rawFile);
						break;

						case "scanInfo":
							if(indexST==0){indexST=1;}
							GetScanInformation(rawFile, indexST);
						break;

						case "scanData":
							if(indexST==0){indexST=1;}
							switch(scanType)
							{
								case "profile":
									GetProfileScanData(rawFile, indexST);
									break;
								case "centroid":
									GetCentroidScanData(rawFile, indexST);
									break;
							}
						break;

						case "chromatogram":
							GetMassChromatogram(rawFile,traceType,mzFilter,mzAcc,massRange);
						break;
					}

		    		    
                }
				else
				{
					Console.WriteLine("No ha especificado la información que se debe descomprimir");
					return;
				}
		    




                Console.WriteLine("Closing " + filename);

                rawFile.Dispose();
            }
            catch (Exception ex)
            {
                Console.WriteLine("The RawFileReader library was not found on your system - " + ex.Message);
            }

        }


		/// @public
		/// @brief Retrieve information for the selected tune in the .Raw file
		/// @param rawFile Raw file selected
		/// @param tuneIndex index associated to the available tunes
		private static void GetTuneData(IRawDataPlus rawFile, int tuneIndex)
		{
			int nTunes =rawFile.GetTuneDataCount();
			Console.WriteLine("========================================================");
			Console.WriteLine("Tune {0} from {1} tunes",tuneIndex+1,nTunes);
			Console.WriteLine("========================================================");
			HeaderItem[] tuneDataHeaders = rawFile.GetTuneDataHeaderInformation();
			TuneDataValues tuneDataValues = rawFile.GetTuneDataValues(tuneIndex, false);
			var stuneDataValues = tuneDataValues.Values;
			
			for (int i = 0; i < tuneDataHeaders.Length; i++)
			{
			Console.WriteLine("Tune parameter: " + tuneDataHeaders[i].Label + " " + stuneDataValues[i]);
			}
		}


		/// @public
		/// @brief Devuelve información asociada a los metodos instrumentales definidos en el .Raw file
		private static void GetInstrumentalMethods(IRawDataPlus rawFile)
		{
			int nMethods =rawFile.InstrumentMethodsCount;
			for (int i = 0; i < nMethods; i++)
			{
			Console.WriteLine("========================================================");
			Console.WriteLine("Instrument method {0} from {1}", i+1,nMethods);
			Console.WriteLine("========================================================");
			var methodInformation = rawFile.GetInstrumentMethod(i);
			Console.WriteLine(methodInformation);
			Console.WriteLine();
			}
		}

		private static void GetSampleInformation(IRawDataPlus rawFile)
		{
			Console.WriteLine("========================================================");
			Console.WriteLine("Raw file and sample information");
			Console.WriteLine("========================================================");
			Console.WriteLine();
			Console.WriteLine("========== Raw file information===============");
					Console.WriteLine("   RAW file: " + rawFile.FileName);
					Console.WriteLine("   RAW file version: " + rawFile.FileHeader.Revision);
					Console.WriteLine("   Creation date: " + rawFile.FileHeader.CreationDate);
					Console.WriteLine("   Operator: " + rawFile.FileHeader.WhoCreatedId);
					Console.WriteLine("   Number of instruments: " + rawFile.InstrumentCount);
					Console.WriteLine("   Description: " + rawFile.FileHeader.FileDescription);
					Console.WriteLine("   Instrument model: " + rawFile.GetInstrumentData().Model);
					Console.WriteLine("   Instrument name: " + rawFile.GetInstrumentData().Name);
					Console.WriteLine("   Serial number: " + rawFile.GetInstrumentData().SerialNumber);
					Console.WriteLine("   Software version: " + rawFile.GetInstrumentData().SoftwareVersion);
					Console.WriteLine("   Firmware version: " + rawFile.GetInstrumentData().HardwareVersion);
					Console.WriteLine("   Units: " + rawFile.GetInstrumentData().Units);
					Console.WriteLine("   Mass resolution: {0:F3} ", rawFile.RunHeaderEx.MassResolution);
					Console.WriteLine("   Number of scans: {0}", rawFile.RunHeaderEx.SpectraCount);
					Console.WriteLine("   Number of scans events: {0}", rawFile.RunHeaderEx.TrailerScanEventCount);
					Console.WriteLine("   Scan range: {0} - {1}", rawFile.RunHeaderEx.FirstSpectrum, rawFile.RunHeaderEx.LastSpectrum);
					Console.WriteLine("   Time range: {0:F2} - {1:F2}", rawFile.RunHeaderEx.StartTime, rawFile.RunHeaderEx.EndTime);
					Console.WriteLine("   Mass range: {0:F4} - {1:F4}", rawFile.RunHeaderEx.LowMass, rawFile.RunHeaderEx.HighMass);
					Console.WriteLine("==========Sample information===============");
					Console.WriteLine("   Sample name: " + rawFile.SampleInformation.SampleName);
					Console.WriteLine("   Sample id: " + rawFile.SampleInformation.SampleId);
					Console.WriteLine("   Sample type: " + rawFile.SampleInformation.SampleType);
					Console.WriteLine("   Sample comment: " + rawFile.SampleInformation.Comment);
					Console.WriteLine("   Sample vial: " + rawFile.SampleInformation.Vial);
					Console.WriteLine("   Sample volume: " + rawFile.SampleInformation.SampleVolume);
					Console.WriteLine("   Sample weight: " + rawFile.SampleInformation.SampleWeight);
					Console.WriteLine("   Sample injection volume: " + rawFile.SampleInformation.InjectionVolume);
					Console.WriteLine("   Sample row number: " + rawFile.SampleInformation.RowNumber);
					Console.WriteLine("   Sample dilution factor: " + rawFile.SampleInformation.DilutionFactor);
					Console.WriteLine("   Sample calibration file: " + rawFile.SampleInformation.CalibrationFile);
					Console.WriteLine("   Sample calibration level: " + rawFile.SampleInformation.CalibrationLevel);
					Console.WriteLine("   Sample ISTD amount: " + rawFile.SampleInformation.IstdAmount);
					Console.WriteLine("   Sample barcode: " + rawFile.SampleInformation.Barcode);
					Console.WriteLine("   Sample barcode status: " + rawFile.SampleInformation.BarcodeStatus);
					Console.WriteLine("   Sample instrument method file: " + rawFile.SampleInformation.InstrumentMethodFile);
					Console.WriteLine("   Sample processing method file: " + rawFile.SampleInformation.ProcessingMethodFile);
					Console.WriteLine("   Sample path: " + rawFile.SampleInformation.Path);
					Console.WriteLine("   Sample raw file name: " + rawFile.SampleInformation.RawFileName);
					Console.WriteLine("   Sample user text: " + rawFile.SampleInformation.UserText);
		}


		private static void GetScanInformation(IRawDataPlus rawFile, int scanIndex)
		{
			int nScans =rawFile.RunHeaderEx.SpectraCount;
			Console.WriteLine("========================================================");
			Console.WriteLine("Scan {0} from {1} scans",scanIndex,nScans);
			Console.WriteLine("========================================================");
			
			var scan = Scan.FromFile(rawFile, scanIndex);
			Console.WriteLine("=== {0}th Scan statistics ==== ",scanIndex);
			Console.WriteLine("Total Ion Current: " + scan.ScanStatistics.TIC);
			Console.WriteLine("Scan Low Mass: " + scan.ScanStatistics.LowMass);
			Console.WriteLine("Scan High Mass: " + scan.ScanStatistics.HighMass);
			Console.WriteLine("Start time: " + scan.ScanStatistics.StartTime);
			Console.WriteLine("Scan Number: " + scan.ScanStatistics.ScanNumber);
			Console.WriteLine("Base Peak Intensity: " + scan.ScanStatistics.BasePeakIntensity);
			Console.WriteLine("Base Peak Mass: " + scan.ScanStatistics.BasePeakMass);
			Console.WriteLine("Scan Mode: " + scan.ScanStatistics.ScanType);

			var scanIndexFilter = rawFile.GetFilterForScanNumber(scanIndex);
			var scanIndexEvent = rawFile.GetScanEventForScanNumber(scanIndex);
			Console.WriteLine("scanFilter.MSOrder: " + scanIndexFilter.MSOrder);

			// var scanReaction = scanIndexEvent.GetReaction(scanIndex);
			// Console.WriteLine("Isolation width: {0:F5} ",scanReaction.IsolationWidth);
			
			var scanTrailerData = rawFile.GetTrailerExtraInformation(scanIndex);
			var scanTrailerDataValues = scanTrailerData.Values;

			for (int i = 0; i < scanTrailerData.Labels.Length; i++)
			{
			Console.WriteLine(scanTrailerData.Labels[i] + " " + scanTrailerDataValues[i]);
			}
			
		}

		/// @public
		/// @brief Devuelve datos profile del espectro de masas asociado a un scan seleccionado dentro de un .Raw file
		/// @param scanIndex indice asociado al scan seleccionado del conjuto de scans disponibles
		private static void GetProfileScanData(IRawDataPlus rawFile, int scanIndex)
		{

			int nScans =rawFile.RunHeaderEx.SpectraCount;
			var scanStatistics = rawFile.GetScanStatsForScanNumber(scanIndex);
			var segmentedScan = rawFile.GetSegmentedScanFromScanNumber(scanIndex, scanStatistics);
			var scanAdvPackedData = rawFile.GetAdvancedPacketData(scanIndex);

			Console.WriteLine("========================================================");
			Console.WriteLine("Profile data for scan {0} from {1} scans. Num. of data: {2}",scanIndex,nScans,segmentedScan.Positions.Length);
			Console.WriteLine("========================================================");
			
			for (int i = 0; i < scanAdvPackedData.NoiseData.Length; i++)
			{
			var scanNoise = scanAdvPackedData.NoiseData[i];	
			Console.WriteLine("Data {0}, mass: {1}, noise: {2}, baseline: {3}",i,scanNoise.Mass,scanNoise.Noise,scanNoise.Baseline);
			}
			Console.WriteLine();
			// Console.WriteLine("Spectrum (normal data) {0} - {1} points", scanIndex, segmentedScan.Positions.Length);


			for (int i = 0; i < segmentedScan.Positions.Length; i++)
			{
			Console.WriteLine("{0}, {1:F6}, {2:F6}", i, segmentedScan.Positions[i], segmentedScan.Intensities[i]);
			}

		}

		/// @public
		/// @brief Devuelve datos centroide del espectro de masas asociado a un scan seleccionado dentro de un .Raw file
		/// @param scanIndex indice asociado al scan seleccionado del conjuto de scans disponibles
		private static void GetCentroidScanData(IRawDataPlus rawFile, int scanIndex)
		{
			int nScans =rawFile.RunHeaderEx.SpectraCount;
			var scanAdvPackedData = rawFile.GetAdvancedPacketData(scanIndex);
			var scanCentroidData = scanAdvPackedData.CentroidData;

			Console.WriteLine("========================================================");
			Console.WriteLine("Centroid data for scan {0} from {1} scans. Num. of data: {2}",scanIndex,nScans,scanCentroidData.Masses.Length);
			Console.WriteLine("========================================================");
			Console.WriteLine();
			Console.WriteLine("No., m/z, Intensity, Resolution, Charge, Baseline, Noise");
			for (int i = 0; i < scanCentroidData.Masses.Length; i++)
			{
			Console.WriteLine("{0}, {1:F6}, {2:F6}, {3}, {4}, {5:F6}, {6:F6} ",i,scanCentroidData.Masses[i],
					scanCentroidData.Intensities[i], scanCentroidData.Resolutions[i], scanCentroidData.Charges[i],
					scanCentroidData.Baselines[i], scanCentroidData.Noises[i]);
			}	    
		}

		/// @public
		/// @brief Retrive information about analysis and scan experiment store into the .Raw file
		private static void GetExperimentsInfo(IRawDataPlus rawFile)
		{

			Console.WriteLine("========================================================");
			Console.WriteLine("File: {0}",rawFile.SampleInformation.RawFileName);
			//Console.WriteLine("File: {0}",rawFile.FileName);
			Console.WriteLine("========================================================");
			var instrumentsName=rawFile.GetAllInstrumentFriendlyNamesFromInstrumentMethod();
				
			//rawFile.ExportInstrumentMethod("./",true);

				int msIndex=0;
				if(rawFile.InstrumentMethodsCount>1){
					//int chromatographIndex=0;		
					msIndex=1;
				}

				var deviceType=rawFile.GetInstrumentType(msIndex);
				var methodInformation = rawFile.GetInstrumentMethod(msIndex);

				int beginStr=methodInformation.IndexOf("Experiment");
				int endStr=methodInformation.IndexOf("Setup");

				if(beginStr>0)
				{
					Console.WriteLine(methodInformation.Substring(beginStr,endStr-beginStr));
				}
				else{
					Console.WriteLine("MS method not defined");
				}
			Console.WriteLine();
		}

		/// @public
		/// @brief Retrive information about analysis and scan experiment store into the .Raw file
		private static void GetRawAnalysisInfo(IRawDataPlus rawFile)
		{			
			Console.WriteLine("File: {0}",rawFile.SampleInformation.RawFileName);
			Console.WriteLine("Creation date: " + rawFile.FileHeader.CreationDate);
			Console.WriteLine("Number of scans: {0}", rawFile.RunHeaderEx.SpectraCount);
			Console.WriteLine("Time range: {0:F2} - {1:F2}", rawFile.RunHeaderEx.StartTime, rawFile.RunHeaderEx.EndTime);
			Console.WriteLine("Mass range: {0:F4} - {1:F4}", rawFile.RunHeaderEx.LowMass, rawFile.RunHeaderEx.HighMass);
		
			//rawFile.ExportInstrumentMethod("./",true);
			int msIndex=0;
			if(rawFile.InstrumentMethodsCount>1)
			{
				msIndex=1;
			}
			var methodInformation = rawFile.GetInstrumentMethod(msIndex);

			// Console.WriteLine("-----> {0}",methodInformation.Contains("Experiment"));
			// return;

			if( !methodInformation.Contains("Experiment") )
			{
				Console.WriteLine("Mode:");
				return;
			}
			string titleExp="Experiment";
			int titleOffset=10;
			if(methodInformation.Contains("Experiments"))
			{
				titleExp="Experiments";
				titleOffset=11;
			}

			int beginStr=methodInformation.IndexOf(titleExp)+titleOffset;
			int endStr=methodInformation.IndexOf("General");

			if(endStr>beginStr)
			{
				string adqMode=methodInformation.Substring(beginStr,endStr-beginStr).Trim();
				
				beginStr=methodInformation.IndexOf("Spectrum data type")+20;
				endStr=methodInformation.IndexOf("Resolution")+10;
				endStr=methodInformation.IndexOf("Resolution",endStr);
					
				string ddMode=string.Empty;
					
				if( methodInformation.Contains("Confirmation") && (methodInformation.IndexOf("Confirmation")-methodInformation.IndexOf("dd-MS")<76))
				{
					ddMode="dd-MS2 Confirmation";
				}

				if( methodInformation.Contains("Discovery") && (methodInformation.IndexOf("Discovery")-methodInformation.IndexOf("dd-MS")<76))
				{
					ddMode="dd-MS2 Discovery";
				}

				string prmTarg=string.Empty;
				if(methodInformation.Contains("Targeted-MS"))
				{
					prmTarg="Targeted-MS2";
				}

				
				adqMode=adqMode + " " + " " + ddMode;
				string tmpStr=string.Empty;
				if(endStr>beginStr)
				{
					tmpStr=methodInformation.Substring(beginStr,endStr-beginStr).Trim();
					tmpStr=tmpStr.Replace("Profile","").Replace("Centroid","").Trim();
					endStr=tmpStr.IndexOf("\n");

					if(endStr>0)
					{
						tmpStr=tmpStr.Substring(0,endStr).Trim();
					}

					adqMode=adqMode + " / "+tmpStr;
				}
				adqMode=adqMode+" "+prmTarg;		
				Console.WriteLine("Mode: {0}",adqMode);
				GetFiltersInfo(rawFile,false);
			}
			else{
				Console.WriteLine("MS method not defined");
			}
		}


		private static void GetFiltersInfo(IRawDataPlus rawFile, bool showFilterParams)
		{
					var scanFilters=rawFile.GetFilters();
					int numberFilters = rawFile.GetFilters().Count;
					var autoFilters = rawFile.GetAutoFilters();

					for(int i=0;i<numberFilters;i++)
					{
						Console.WriteLine("Filter {0}: {1}",i+1,autoFilters[i].ToString());					
						if(showFilterParams)
						{
							Console.WriteLine("=====================================================");
							Console.WriteLine("Accurate mass: {0}",scanFilters[i].AccurateMass);
							Console.WriteLine("Compensation Voltage: {0}",scanFilters[i].CompensationVoltage);
							Console.WriteLine("Compensation Voltage Count: {0}",scanFilters[i].CompensationVoltageCount);
							Console.WriteLine("Compensation Voltage Type: {0}",scanFilters[i].CompensationVoltType);
							Console.WriteLine("Corona: {0}",scanFilters[i].Corona);
							Console.WriteLine("Dependent: {0}",scanFilters[i].Dependent);
							Console.WriteLine("Detector: {0}",scanFilters[i].Detector);
							Console.WriteLine("Detector Value: {0}",scanFilters[i].DetectorValue);
							Console.WriteLine("Electron Capture Dissociation: {0}",scanFilters[i].ElectronCaptureDissociation);
							Console.WriteLine("Electron Capture Dissociation Value: {0}",scanFilters[i].ElectronCaptureDissociationValue);
							Console.WriteLine("Electron Transfer Dissociation: {0}",scanFilters[i].ElectronTransferDissociation);
							Console.WriteLine("Electron Transfer Dissociation Value: {0}",scanFilters[i].ElectronTransferDissociationValue);
							Console.WriteLine("Enhanced: {0}",scanFilters[i].Enhanced);
							Console.WriteLine("Field Free Region: {0}",scanFilters[i].FieldFreeRegion);
							Console.WriteLine("Higher Energy CiD: {0}",scanFilters[i].HigherEnergyCiD);
							Console.WriteLine("Higher Energy CiD Value: {0}",scanFilters[i].HigherEnergyCiDValue);
							Console.WriteLine("Ionization Mode: {0}",scanFilters[i].IonizationMode);
							Console.WriteLine("Local Name: {0}",scanFilters[i].LocaleName);
							Console.WriteLine("Lock: {0}",scanFilters[i].Lock);
							Console.WriteLine("Mass Analyzer: {0}",scanFilters[i].MassAnalyzer);
							Console.WriteLine("Mass Count: {0}",scanFilters[i].MassCount);
							Console.WriteLine("Mass Precision: {0}",scanFilters[i].MassPrecision);
							Console.WriteLine("Mass Range Count: {0}",scanFilters[i].MassRangeCount);
							Console.WriteLine("Meta Filters: {0}",scanFilters[i].MetaFilters);
							Console.WriteLine("MS Order: {0}",scanFilters[i].MSOrder);
							Console.WriteLine("Multi Notch: {0}",scanFilters[i].MultiNotch);
							Console.WriteLine("Multiple Photon Dissociation: {0}",scanFilters[i].MultiplePhotonDissociation);
							Console.WriteLine("Multiple Photon Dissociation Value: {0}",scanFilters[i].MultiplePhotonDissociationValue);
							Console.WriteLine("Multiplex: {0}",scanFilters[i].Multiplex);
							Console.WriteLine("Multi State Activation {0}",scanFilters[i].MultiStateActivation);
							Console.WriteLine("Name: {0}",scanFilters[i].Name);
							Console.WriteLine("Parameter A: {0}",scanFilters[i].ParamA);
							Console.WriteLine("Parameter B: {0}",scanFilters[i].ParamB);
							Console.WriteLine("Parameter F: {0}",scanFilters[i].ParamF);
							Console.WriteLine("Parameter R: {0}",scanFilters[i].ParamR);
							Console.WriteLine("Parameter V: {0}",scanFilters[i].ParamV);
							Console.WriteLine("PhotoIonization: {0}",scanFilters[i].PhotoIonization);
							Console.WriteLine("Polarity: {0}",scanFilters[i].Polarity);
							Console.WriteLine("Pulsed Q Dissociation: {0}",scanFilters[i].PulsedQDissociation);
							Console.WriteLine("Pulsed Q Dissociation Value: {0}",scanFilters[i].PulsedQDissociationValue);
							Console.WriteLine("Scan Data: {0}",scanFilters[i].ScanData);
							Console.WriteLine("Scan Mode: {0}",scanFilters[i].ScanMode);
							Console.WriteLine("Scan Type Index: {0}",scanFilters[i].ScanTypeIndex);
							Console.WriteLine("Scan Sector: {0}",scanFilters[i].SectorScan);
							Console.WriteLine("Source Fragmentation Value Count: {0}",scanFilters[i].SouceFragmentaionValueCount);
							Console.WriteLine("Source Fragmentation: {0}",scanFilters[i].SourceFragmentation);
							Console.WriteLine("Source Fragmentation Info Count: {0}",scanFilters[i].SourceFragmentationInfoCount);
							Console.WriteLine("SourceFragmentationInfoValid Length: {0}",scanFilters[i].SourceFragmentationInfoValid.Length);
							Console.WriteLine("Source Fragmentation Type: {0}",scanFilters[i].SourceFragmentationType);
							Console.WriteLine("Supplemental Activation: {0}",scanFilters[i].SupplementalActivation);
							Console.WriteLine("Turbo Scan: {0}",scanFilters[i].TurboScan);
							Console.WriteLine("Ultra: {0}",scanFilters[i].Ultra);
							Console.WriteLine("Unique Mass Count: {0}",scanFilters[i].UniqueMassCount);
							Console.WriteLine("Wideband: {0}",scanFilters[i].Wideband);
						}
					}
		}

		private static void GetFilterInfoForScan(IRawDataPlus rawFile, int scanIndex)
		{
					var scanFilter = rawFile.GetFilterForScanNumber(scanIndex);
					Console.WriteLine("   Scan filter ({0}): {1} ", scanIndex,scanFilter.ToString());
		}


		private static void GetEventsInfo(IRawDataPlus rawFile)
		{
			int initScan=rawFile.RunHeaderEx.FirstSpectrum;
			int lastScan=rawFile.RunHeaderEx.LastSpectrum;
			for (int i = initScan; i <= lastScan; i++)
			{

					var scanFilter = rawFile.GetFilterForScanNumber(i);
					var scan = Scan.FromFile(rawFile, i);
					Console.WriteLine(" {0};{1:F10};{2};{3:F4};{4:F6};{5:F6};{6:F6};{7:F6}", i,scan.ScanStatistics.StartTime,
						scanFilter.ToString(),scan.ScanStatistics.TIC,
						scan.ScanStatistics.LowMass,scan.ScanStatistics.HighMass,
						scan.ScanStatistics.BasePeakIntensity,scan.ScanStatistics.BasePeakMass);
			}
		}



		/// @public
		/// @brief Retrive information on chromatographic elution method
		/// @param rawFile .Raw file to analyze
		private static void GetElutionMethod(IRawDataPlus rawFile)
		{
			int chromMethodIndex=0;
			var methodInformation = rawFile.GetInstrumentMethod(chromMethodIndex);

			if (methodInformation.Contains("Chromatography"))
			{
				Console.WriteLine(methodInformation);
				Console.WriteLine();
			}
		}

        private static void GetMassChromatogram(IRawDataPlus rawFile, string traceType, string filterMS, double tol,
				double[] massRange)
        {
            var nInsts = rawFile.GetInstrumentCountOfType(Device.MS);
			double lowMass=massRange[0];
			double highMass=massRange[1];
			if(lowMass<0){lowMass=rawFile.RunHeader.LowMass;}
			if(highMass<0){highMass=rawFile.RunHeader.HighMass;}

            if (nInsts > 0)
            {
                rawFile.SelectInstrument(Device.MS, 1);
				var settings = new ChromatogramTraceSettings();
				MassOptions tolerance = new MassOptions(){
						Tolerance=tol,
						ToleranceUnits= ToleranceUnits.ppm
					};

				

				switch(traceType){
					case "basePeak":
                		settings.Trace=	TraceType.BasePeak;
						settings.Filter=filterMS;
                		settings.MassRangeCount = 1;
					break;

					case "specMax":
                		settings.Trace=	TraceType.SpectrumMax;
						settings.Filter=filterMS;
						settings.MassRangeCount = 1;
					break;

					case "tic":
                		settings.Trace=	TraceType.TIC;
						settings.Filter=filterMS;
					break;

					case "massRange":
						settings.Trace=	TraceType.MassRange;
						settings.Filter=filterMS;
						settings.MassRangeCount = 1;
               		break;

					default:
                		settings.Trace=	TraceType.BasePeak;
						settings.Filter=filterMS;
                		settings.MassRangeCount = 1;
					break;
				}
				settings.SetMassRange(0, new ThermoFisher.CommonCore.Data.Business.Range(lowMass,highMass));
				


                try
                {
					IChromatogramData data;
                    if(tol>0.0){
						data = rawFile.GetChromatogramData(new IChromatogramSettings[] { settings }, 
						-1, -1,tolerance);
					}else{
						data = rawFile.GetChromatogramData(new IChromatogramSettings[] { settings }, 
						-1, -1);
					}




                    var trace = ChromatogramSignal.FromChromatogramData(data);

                    if (trace[0].Length > 0)
                    {

                        for (int i = 0; i < trace[0].Length; i++)
                        {
                            Console.WriteLine("{0:F10};{1:F6}", trace[0].Times[i], trace[0].Intensities[i]);
                        }
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine("The RawFileReader library was not found on your system - " + ex.Message);
                }
            }
        }


	
    }    
}
