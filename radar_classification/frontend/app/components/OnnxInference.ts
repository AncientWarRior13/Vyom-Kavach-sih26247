
let ort: any = null;

export interface EnsembleDetails {
  probabilityQuantum: number;
  probabilityClassical: number;
  metaCoefficients: { quantum: number; classical: number; bias: number };
  dominantModel: string;
}

export interface InferenceResult {
  predictedClass: 'bird' | 'drone' | 'aircraft' | 'anomaly';
  confidences: Record<'bird' | 'drone' | 'aircraft' | 'anomaly', number>;
  probabilities: number[];
  inferenceTime: number;
  mode?: string;
  useEnsemble?: boolean;
  ensembleDetails?: EnsembleDetails;
  isStub?: boolean;
  confidencesUnavailable?: boolean;
}

export class OnnxModelManager {
  private session: any = null;
  private isLoaded: boolean = false;
  private classes: ('aircraft' | 'anomaly' | 'bird' | 'drone')[] = ['aircraft', 'anomaly', 'bird', 'drone'];
  private runLock: Promise<any> = Promise.resolve();
  private expectedFeatures: number = 31; 

  async loadModelFromBuffer(buffer: ArrayBuffer): Promise<boolean> {
    try {
      if (typeof window === 'undefined') return false;

      
      if (!ort) {
        ort = await import('onnxruntime-web');
      }

      
      this.session = await ort.InferenceSession.create(new Uint8Array(buffer));
      this.isLoaded = true;
      console.log("ONNX model loaded successfully!");

      
      if (this.session.inputs && this.session.inputs.length > 0) {
        const dims = this.session.inputs[0].dims;
        if (dims && dims.length > 0) {
          const lastDim = dims[dims.length - 1];
          const numFeats = typeof lastDim === 'number' ? lastDim : parseInt(lastDim, 10);
          if (!isNaN(numFeats)) {
            this.expectedFeatures = numFeats;
            console.log(`ONNX Model detected. Expected input features: ${numFeats}`);
          }
        }
      }
      return true;
    } catch (error) {
      console.error("Failed to load ONNX model:", error);
      this.isLoaded = false;
      throw error;
    }
  }

  isModelLoaded(): boolean {
    return this.isLoaded;
  }

  unloadModel() {
    this.session = null;
    this.isLoaded = false;
    this.expectedFeatures = 31;
  }

  async predict(
    features31: number[],
    features11: number[],
    groundTruth: 'bird' | 'drone' | 'aircraft' | 'anomaly',
    snrDb: number,
    useQuantum: boolean = false,
    useEnsemble: boolean = false
  ): Promise<InferenceResult> {
    if (useQuantum) {
      
      let baseResult: InferenceResult;
      if (this.isLoaded && this.session) {
        try {
          baseResult = await this.predict(features31, features11, groundTruth, snrDb, false, false);
        } catch (e) {
          baseResult = this.runStubInference(features11, groundTruth, snrDb);
        }
      } else {
        baseResult = this.runStubInference(features11, groundTruth, snrDb);
      }

      try {
        const apiBase = process.env.NEXT_PUBLIC_API_BASE || process.env.NEXT_PUBLIC_QUANTUM_API_BASE || 'http://localhost:8081';
        const response = await fetch(`${apiBase}/predict`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            doppler_velocity: features11[0],
            rcs_estimate: features11[2],
            micro_doppler_bandwidth: features11[3],
            micro_doppler_jitter: features11[4],
            use_ensemble: useEnsemble,
            mode: useEnsemble ? 'ensemble' : 'quantum'
          })
        });

        if (!response.ok) {
          throw new Error(`Quantum API responded with status ${response.status}`);
        }

        const data = await response.json();
        const pDrone = data.probability_drone ?? 0.0;
        const pNotDrone = data.probability_not_drone ?? 1.0;
        const inferenceTime = data.inference_latency_ms ?? 0.0;

        const confidences: Record<'bird' | 'drone' | 'aircraft' | 'anomaly', number> = {
          bird: 0,
          drone: pDrone,
          aircraft: 0,
          anomaly: 0
        };

        
        const otherClasses: ('bird' | 'aircraft' | 'anomaly')[] = ['bird', 'aircraft', 'anomaly'];
        const sumOthers = otherClasses.reduce((sum, cls) => sum + (baseResult.confidences[cls] || 0), 0);

        otherClasses.forEach(cls => {
          const prop = sumOthers > 0 ? (baseResult.confidences[cls] || 0) / sumOthers : 1 / 3;
          confidences[cls] = prop * pNotDrone;
        });

        
        let predictedClass: 'bird' | 'drone' | 'aircraft' | 'anomaly';
        if (data.prediction === 'drone') {
          predictedClass = 'drone';
        } else {
          let maxVal = -1;
          let bestClass = otherClasses[0];
          otherClasses.forEach(cls => {
            if (confidences[cls] > maxVal) {
              maxVal = confidences[cls];
              bestClass = cls;
            }
          });
          predictedClass = bestClass;
        }

        const probabilities = [
          confidences.bird,
          confidences.drone,
          confidences.aircraft,
          confidences.anomaly
        ];

        let ensembleDetails: EnsembleDetails | undefined = undefined;
        if (data.ensemble_details) {
          ensembleDetails = {
            probabilityQuantum: data.ensemble_details.probability_quantum ?? 0,
            probabilityClassical: data.ensemble_details.probability_classical ?? 0,
            metaCoefficients: {
              quantum: data.ensemble_details.meta_coefficients?.quantum ?? 0,
              classical: data.ensemble_details.meta_coefficients?.classical ?? 0,
              bias: data.ensemble_details.meta_coefficients?.bias ?? 0,
            },
            dominantModel: data.ensemble_details.dominant_model ?? 'quantum',
          };
        }

        return {
          predictedClass,
          confidences,
          probabilities,
          inferenceTime,
          mode: data.mode ?? (useEnsemble ? 'ensemble' : 'quantum'),
          useEnsemble: !!useEnsemble,
          ensembleDetails,
          isStub: false
        };
      } catch (err) {
        console.error("Quantum prediction failed, falling back to classical:", err);
        return baseResult;
      }
    }

    
    try {
      const apiBase = process.env.NEXT_PUBLIC_API_BASE || process.env.NEXT_PUBLIC_QUANTUM_API_BASE || 'http://localhost:8081';
      let endpoint = `${apiBase}/predict/classic`;
      let payload: Record<string, number> = {};

      const feat11Names = [
        "doppler_velocity", "doppler_freq_shift", "rcs_estimate",
        "micro_doppler_bandwidth", "micro_doppler_jitter", "range_m",
        "azimuth_deg", "altitude_m", "velocity_variance", "acceleration", "snr_db"
      ];

      if (this.expectedFeatures === 31) {
        endpoint = `${apiBase}/predict/31`;
        const feat31Names = [
          "snr_db",
          "amplitude_mean", "amplitude_std", "amplitude_min", "amplitude_max",
          "amplitude_range", "amplitude_energy", "amplitude_zcr",
          "amplitude_dominant_freq", "amplitude_spectral_bandwidth", "amplitude_diff_var",
          "velocity_mean", "velocity_std", "velocity_min", "velocity_max",
          "velocity_range", "velocity_energy", "velocity_zcr",
          "velocity_dominant_freq", "velocity_spectral_bandwidth", "velocity_diff_var",
          "energy_mean", "energy_std", "energy_min", "energy_max",
          "energy_range", "energy_energy", "energy_zcr",
          "energy_dominant_freq", "energy_spectral_bandwidth", "energy_diff_var"
        ];
        feat31Names.forEach((name, idx) => {
          payload[name] = features31[idx] ?? 0;
        });
      } else if (this.expectedFeatures === 41 || this.expectedFeatures === 42) {
        endpoint = `${apiBase}/predict/42`;
        const feats41 = [...features11, ...features31.slice(1)];
        const feat42Names = [
          ...feat11Names,
          "amplitude_mean", "amplitude_std", "amplitude_min", "amplitude_max",
          "amplitude_range", "amplitude_energy", "amplitude_zcr",
          "amplitude_dominant_freq", "amplitude_spectral_bandwidth", "amplitude_diff_var",
          "velocity_mean", "velocity_std", "velocity_min", "velocity_max",
          "velocity_range", "velocity_energy", "velocity_zcr",
          "velocity_dominant_freq", "velocity_spectral_bandwidth", "velocity_diff_var",
          "energy_mean", "energy_std", "energy_min", "energy_max",
          "energy_range", "energy_energy", "energy_zcr",
          "energy_dominant_freq", "energy_spectral_bandwidth", "energy_diff_var"
        ];
        feat42Names.forEach((name, idx) => {
          payload[name] = feats41[idx] ?? 0;
        });
      } else {
        feat11Names.forEach((name, idx) => {
          payload[name] = features11[idx] ?? 0;
        });
      }

      const start = performance.now();
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (res.ok) {
        const data = await res.json();
        const inferenceTime = performance.now() - start;
        const conf = data.probabilities || {};
        const confidences: Record<'bird' | 'drone' | 'aircraft' | 'anomaly', number> = {
          aircraft: conf.aircraft ?? 0,
          anomaly: conf.anomaly ?? 0,
          bird: conf.bird ?? 0,
          drone: conf.drone ?? 0
        };
        const rawLabel = String(data.label || 'aircraft');
        const predictedClass: 'aircraft' | 'anomaly' | 'bird' | 'drone' =
          (rawLabel === 'stealth_uav') ? 'anomaly' : (rawLabel as any);
        const probs = [confidences.aircraft, confidences.anomaly, confidences.bird, confidences.drone];

        return {
          predictedClass,
          confidences,
          probabilities: probs,
          inferenceTime,
          isStub: false,
          confidencesUnavailable: false
        };
      }
    } catch (apiErr) {
      
    }

    if (this.isLoaded && this.session) {
      
      const currentRun = this.runLock.then(async () => {
        
        const feats41 = [...features11, ...features31.slice(1)]; 

        let primaryFeatures: number[];
        if (this.expectedFeatures === 11) {
          primaryFeatures = features11;
        } else if (this.expectedFeatures === 41 || this.expectedFeatures === 42) {
          primaryFeatures = feats41;
        } else {
          primaryFeatures = features31;
        }

        try {
          return await this.runInferenceForFeatures(primaryFeatures, groundTruth, snrDb);
        } catch (e) {
          console.warn(`ONNX primary inference failed with ${primaryFeatures.length} features, attempting fallbacks:`, e);

          
          const candidates = [features11, features31, feats41].filter(
            c => c.length !== primaryFeatures.length
          );

          for (const cand of candidates) {
            try {
              const result = await this.runInferenceForFeatures(cand, groundTruth, snrDb);
              this.expectedFeatures = cand.length;
              return result;
            } catch (fallbackErr) {
              
            }
          }
          console.error("All ONNX feature formats failed, falling back to stub.");
          throw new Error("ONNX inference failed across all feature configurations.");
        }
      });

      this.runLock = currentRun.catch(() => { });

      try {
        return await currentRun;
      } catch (err) {
        
      }
    }

    
    return this.runStubInference(features11, groundTruth, snrDb);
  }

  private async runInferenceForFeatures(
    features: number[],
    groundTruth: 'bird' | 'drone' | 'aircraft' | 'anomaly',
    snrDb: number
  ): Promise<InferenceResult> {
    const numFeats = features.length;
    const inputTensor = new ort.Tensor('float32', new Float32Array(features), [1, numFeats]);

    const inputNames = this.session.inputNames;
    const outputNames = this.session.outputNames;
    const feeds: Record<string, any> = {};
    const inputKey = (inputNames && inputNames.includes('features')) ? 'features' : (inputNames[0] || 'features');
    feeds[inputKey] = inputTensor;

    const start = performance.now();

    
    let results: any;
    let hasProbs = true;

    try {
      results = await this.session.run(feeds, outputNames);
    } catch (err: any) {
      const errMsg = err?.message || '';
      if (
        errMsg.includes('non-tensor') ||
        errMsg.includes('index 1') ||
        errMsg.includes('value is not supported') ||
        errMsg.includes('Reading data')
      ) {
        console.warn("Sequence-of-maps output not supported by browser. Falling back to label-only inference.");
        results = await this.session.run(feeds, [outputNames[0]]);
        hasProbs = false;
      } else {
        throw err;
      }
    }

    const inferenceTime = performance.now() - start;

    
    const classes = ['aircraft', 'anomaly', 'bird', 'drone'] as const;

    
    const outputTensor = results && outputNames[0] ? results[outputNames[0]] : null;
    let labelClass: 'aircraft' | 'anomaly' | 'bird' | 'drone' | null = null;
    if (outputTensor && outputTensor.data && outputTensor.data.length > 0) {
      const val = String(outputTensor.data[0]);
      const mappedVal = (val === 'stealth_uav') ? 'anomaly' : val;
      if (['aircraft', 'anomaly', 'bird', 'drone'].includes(mappedVal)) {
        labelClass = mappedVal as any;
      } else {
        const idx = Number(val);
        if (!isNaN(idx) && classes[idx]) {
          labelClass = classes[idx];
        }
      }
    }

    
    let confidences: Record<'bird' | 'drone' | 'aircraft' | 'anomaly', number> = {
      aircraft: 0, anomaly: 0, bird: 0, drone: 0
    };

    if (hasProbs && outputNames[1] && results[outputNames[1]]) {
      const probTensor = results[outputNames[1]];
      try {
        if (Array.isArray(probTensor.data) || probTensor.data instanceof Float32Array) {
          const probsData = probTensor.data as Float32Array;
          confidences = {
            aircraft: probsData[0] ?? 0,
            anomaly: probsData[1] ?? 0,
            bird: probsData[2] ?? 0,
            drone: probsData[3] ?? 0
          };
        } else if (typeof probTensor.data === 'object') {
          const mapData = probTensor.data[0] || probTensor.data;
          confidences = {
            aircraft: mapData['aircraft'] ?? mapData[0] ?? 0,
            anomaly: mapData['anomaly'] ?? mapData['stealth_uav'] ?? mapData[1] ?? 0,
            bird: mapData['bird'] ?? mapData[2] ?? 0,
            drone: mapData['drone'] ?? mapData[3] ?? 0
          };
        }
      } catch (parseErr) {
        console.warn("Failed to parse probabilities tensor data:", parseErr);
        hasProbs = false;
      }
    } else {
      hasProbs = false;
    }

    let predictedClass: 'aircraft' | 'anomaly' | 'bird' | 'drone';

    if (hasProbs) {
      
      const DRONE_THRESHOLD = 0.54;
      if (confidences.drone >= DRONE_THRESHOLD) {
        predictedClass = 'drone';
      } else {
        const otherClasses: ('aircraft' | 'anomaly' | 'bird')[] = ['aircraft', 'anomaly', 'bird'];
        let maxVal = -1;
        let bestClass: 'aircraft' | 'anomaly' | 'bird' = 'aircraft';
        otherClasses.forEach(cls => {
          if (confidences[cls] > maxVal) {
            maxVal = confidences[cls];
            bestClass = cls;
          }
        });
        predictedClass = bestClass;
      }
    } else {
      
      predictedClass = labelClass || 'aircraft';
      const snrRatio = Math.max(0, Math.min(1, (snrDb - (-25.0)) / 15.0));
      const predictedProb = Math.max(0.55, Math.min(0.98, 0.70 + snrRatio * 0.25 + (Math.random() - 0.5) * 0.04));
      const otherProb = (1.0 - predictedProb) / 3.0;

      confidences = {
        aircraft: otherProb,
        anomaly: otherProb,
        bird: otherProb,
        drone: otherProb
      };
      confidences[predictedClass] = predictedProb;
    }

    const probs = [confidences.aircraft, confidences.anomaly, confidences.bird, confidences.drone];

    return {
      predictedClass,
      confidences,
      probabilities: probs,
      inferenceTime,
      isStub: false,
      confidencesUnavailable: !hasProbs
    };
  }

  private runStubInference(features: number[], groundTruth: 'bird' | 'drone' | 'aircraft' | 'anomaly', snrDb: number): InferenceResult {
    const start = performance.now();
    let birdScore = 0;
    let droneScore = 0;
    let aircraftScore = 0;
    let anomalyScore = 0;

    
    const doppler = Math.abs(features[0] ?? 0);
    const rcs = features[2] ?? 0;
    const microBw = features[3] ?? 0;
    const velVar = features[8] ?? 0;

    if (rcs > 3.0 || doppler > 35.0) {
      aircraftScore += 5.5;
    } else if (microBw > 120.0 || (rcs >= -14.0 && rcs <= 2.0 && doppler >= 2.0 && doppler <= 35.0)) {
      droneScore += 5.0;
    } else if (rcs < -10.0 && microBw < 100.0) {
      birdScore += 4.5;
    } else if (velVar > 5.0) {
      anomalyScore += 4.5;
    } else {
      droneScore += 3.0;
    }

    const snrRatio = Math.max(0.05, Math.min(1.0, (snrDb - (-25.0)) / 15.0));
    const scoreNoise = (1.0 - snrRatio) * 3.5;

    
    birdScore += (Math.random() - 0.5) * scoreNoise;
    droneScore += (Math.random() - 0.5) * scoreNoise;
    aircraftScore += (Math.random() - 0.5) * scoreNoise;
    anomalyScore += (Math.random() - 0.5) * scoreNoise;

    
    const scores = [aircraftScore, anomalyScore, birdScore, droneScore];
    const maxScore = Math.max(...scores);
    const exps = scores.map(s => Math.exp(s - maxScore));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map(e => e / sumExps);

    let maxVal = -1;
    let maxIdx = 0;
    probs.forEach((p, idx) => {
      if (p > maxVal) {
        maxVal = p;
        maxIdx = idx;
      }
    });

    const inferenceTime = performance.now() - start;

    return {
      predictedClass: this.classes[maxIdx],
      confidences: {
        aircraft: probs[0],
        anomaly: probs[1],
        bird: probs[2],
        drone: probs[3]
      },
      probabilities: probs,
      inferenceTime,
      isStub: true,
      mode: 'DEMO MODE - SYNTHETIC STUB'
    };
  }
}
