import { qrngProvider, type QRNGProvider } from './qrng';
import { type GenerationRequest, type TokenResponse } from '@shared/schema';

export interface LayerAnalysis {
  attention: number;
  ffn: number;
  embedding: number;
  [key: string]: number;
}

export class QuantumLLMEngine {
  private qrng: QRNGProvider;
  private tokenCount: number = 0;
  private startTime: number = 0;
  private entropyUsed: number = 0;

  // Vocabulary for token generation - standard English words, no quantum themes
  private vocabulary = [
    "the", "a", "an", "is", "was", "are", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "can", "shall", "to", "of", "in", "for", "on",
    "with", "at", "by", "from", "up", "about", "into", "through", "during", "before",
    "after", "above", "below", "between", "under", "along", "following", "behind", "beyond", "plus",
    "except", "but", "yet", "so", "and", "or", "nor", "if", "then", "because",
    "as", "until", "while", "although", "whether", "since", "unless", "despite", "regarding", "concerning",
    "I", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their", "this", "that",
    "these", "those", "which", "what", "who", "whom", "whose", "where", "when", "why",
    "how", "all", "both", "each", "every", "any", "some", "most", "none", "several",
    "many", "few", "more", "less", "much", "little", "very", "quite", "just", "only",
    "even", "already", "still", "never", "always", "sometimes", "often", "usually", "generally", "specifically",
    "particularly", "especially", "mainly", "mostly", "primarily", "secondly", "thirdly", "finally", "therefore", "however",
    "moreover", "furthermore", "meanwhile", "otherwise", "nevertheless", "nonetheless", "accordingly", "consequently", "thus", "hence"
  ];
  
  // Create probability distribution for vocabulary (simulating logits)
  private baseLogits: number[] = [];

  constructor() {
    this.qrng = qrngProvider;
    // Initialize base logits (uniform distribution)
    this.baseLogits = new Array(this.vocabulary.length).fill(0);
  }

  async *generateStream(request: GenerationRequest): AsyncGenerator<TokenResponse> {
    this.tokenCount = 0;
    this.startTime = Date.now();
    this.entropyUsed = 0;

    const { prompt, profile, maxTokens, temperature } = request;
    
    // CRITICAL: Pre-check QRNG availability - NO FALLBACK ALLOWED
    if (profile !== 'strict') {
      const isAvailable = await this.qrng.isAvailable();
      if (!isAvailable) {
        throw new Error('QRNG unavailable - generation halted. True quantum randomness is required. NO pseudorandom fallback.');
      }
    }
    
    while (this.tokenCount < maxTokens) {
      try {
        // Generate both modified token and vector interpretation
        const { token, vectorInterpretation, rawVector } = await this.generateDualOutput(profile, temperature);
        const layerAnalysis = await this.analyzeLayerActivity(profile);
        const performanceMetrics = this.getPerformanceMetrics();

        // Return both outputs - the QRNG-modified token and the vector interpretation
        yield {
          token,
          influence: `QRNG Vector: [${rawVector.slice(0, 5).map(v => v.toFixed(3)).join(', ')}...] → ${vectorInterpretation}`,
          layerAnalysis,
          performanceMetrics
        };

        this.tokenCount++;
        
        // Variable delay based on profile
        const delay = this.calculateDelay(profile);
        await new Promise(resolve => setTimeout(resolve, delay));
      } catch (error) {
        // If QRNG fails mid-generation, STOP IMMEDIATELY - NO FALLBACK
        console.error('[LLM] CRITICAL: QRNG failure during generation, stopping:', error);
        throw new Error(`Generation halted - quantum randomness unavailable: ${error instanceof Error ? error.message : 'QRNG failure'}`);
      }
    }
  }

  private async generateDualOutput(profile: string, temperature: number): Promise<{
    token: string;
    vectorInterpretation: string;
    rawVector: number[];
  }> {
    if (profile === 'strict') {
      // Deterministic selection - no QRNG modification
      const token = this.vocabulary[this.tokenCount % this.vocabulary.length];
      return {
        token,
        vectorInterpretation: 'deterministic',
        rawVector: [0]
      };
    }

    // Get QRNG vector to modify logits
    const vocabSize = this.vocabulary.length;
    const quantumModifiers = await this.qrng.getRandomFloats(vocabSize, -1, 1);
    
    // Apply quantum modifiers to base logits
    const modifiedLogits = this.baseLogits.map((baseLogit, i) => {
      const modifier = quantumModifiers[i] * this.getModifierStrength(profile);
      return baseLogit + modifier;
    });
    
    // Apply temperature scaling
    const scaledLogits = modifiedLogits.map(logit => logit / temperature);
    
    // Convert logits to probabilities using softmax
    const maxLogit = Math.max(...scaledLogits);
    const expLogits = scaledLogits.map(logit => Math.exp(logit - maxLogit));
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    const probabilities = expLogits.map(exp => exp / sumExp);
    
    // Sample from the probability distribution using QRNG
    const randomValue = await this.qrng.getRandomFloats(1, 0, 1);
    let cumSum = 0;
    let selectedIndex = 0;
    
    for (let i = 0; i < probabilities.length; i++) {
      cumSum += probabilities[i];
      if (randomValue[0] < cumSum) {
        selectedIndex = i;
        break;
      }
    }
    
    // Generate vector interpretation - scale by all 1s and select word
    const interpretationVector = quantumModifiers.map(v => v * 1); // Scale by 1s as requested
    const interpretationIndex = Math.abs(Math.floor(interpretationVector.reduce((a, b) => a + b, 0) * 10)) % this.vocabulary.length;
    const vectorInterpretation = this.vocabulary[interpretationIndex];
    
    this.entropyUsed += vocabSize * 4 + 4; // Track entropy consumption
    
    return {
      token: this.vocabulary[selectedIndex],
      vectorInterpretation,
      rawVector: quantumModifiers
    };
  }



  private getModifierStrength(profile: string): number {
    // Controls how much QRNG modifies the logits
    switch (profile) {
      case 'light': return 0.3;   // Light quantum influence
      case 'medium': return 0.6;  // Moderate quantum influence
      case 'spicy': return 1.0;   // Strong quantum influence
      default: return 0;
    }
  }

  private async getQuantumLogitModifier(profile: string): Promise<number[]> {
    if (profile === 'strict') {
      return new Array(this.vocabulary.length).fill(0);
    }
    
    // Get quantum random vector for logit modification
    const modifierSize = Math.min(10, this.vocabulary.length); // Sample first 10 for display
    const modifiers = await this.qrng.getRandomFloats(modifierSize, -1, 1);
    return modifiers;
  }

  private async analyzeLayerActivity(profile: string): Promise<LayerAnalysis> {
    if (profile === 'strict') {
      return {
        attention: 0.3,
        ffn: 0.3,
        embedding: 0.3
      };
    }

    const randomValues = await this.qrng.getRandomFloats(3, 0.1, 0.9);
    return {
      attention: randomValues[0],
      ffn: randomValues[1],
      embedding: randomValues[2]
    };
  }

  private calculateDelay(profile: string): number {
    const baseDelay = 150; // Base delay in ms
    const variance = profile === 'strict' ? 0 : Math.random() * 100;
    return baseDelay + variance;
  }

  private getPerformanceMetrics() {
    const elapsed = Date.now() - this.startTime;
    const tokensPerSec = this.tokenCount / (elapsed / 1000);
    const avgLatency = this.tokenCount > 0 ? elapsed / this.tokenCount : 0;

    return {
      latency: Math.round(avgLatency),
      tokensPerSec: Number(tokensPerSec.toFixed(1)),
      entropyUsed: this.entropyUsed
    };
  }

  async getQRNGStatus() {
    const isAvailable = await this.qrng.isAvailable();
    const poolStatus = this.qrng instanceof (await import('./qrng')).QuantumBlockchainsQRNG 
      ? (this.qrng as any).getEntropyPoolStatus()
      : { size: 0, percentage: 0 };
    
    return {
      available: isAvailable,
      provider: 'Quantum Blockchains',
      entropyPool: poolStatus
    };
  }
}

export const llmEngine = new QuantumLLMEngine();
