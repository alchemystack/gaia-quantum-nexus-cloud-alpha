import https from 'https';
import fs from 'fs';
import path from 'path';

export interface QRNGProvider {
  getRandomBytes(length: number): Promise<Buffer>;
  getRandomIntegers(count: number, min: number, max: number): Promise<number[]>;
  getRandomFloats(count: number, min: number, max: number): Promise<number[]>;
  isAvailable(): Promise<boolean>;
}

export class QuantumBlockchainsQRNG implements QRNGProvider {
  private apiKey: string;
  private baseUrl: string = 'https://qrng.qbck.io';
  private entropyPool: Buffer[] = [];
  private poolSize: number = 1000; // Buffer pool size in bytes
  private isPooling: boolean = false;

  constructor() {
    this.apiKey = process.env.QBCK_API_KEY || process.env.QRNG_API_KEY || '';
    if (!this.apiKey) {
      console.warn('QRNG API key not found in environment variables. Quantum generation will be unavailable.');
    }
    
    // Start entropy pooling if API key is available
    if (this.apiKey) {
      this.startEntropyPooling();
    }
  }

  private async makeRequest(endpoint: string): Promise<any> {
    return new Promise((resolve, reject) => {
      // Load the SSL certificate chain if available
      let options: https.RequestOptions = {
        hostname: 'qrng.qbck.io',
        path: endpoint,
        method: 'GET',
        headers: {
          'User-Agent': 'Gaia-Quantum-Nexus/1.0'
        }
      };

      // Try to load chain.pem certificate if it exists
      const chainPemPath = path.join(process.cwd(), 'chain.pem');
      if (fs.existsSync(chainPemPath)) {
        options.ca = fs.readFileSync(chainPemPath);
      }

      const req = https.request(options, (res) => {
        let data = '';
        res.on('data', chunk => data += chunk);
        res.on('end', () => {
          try {
            const result = JSON.parse(data);
            if (result.error === 'OK') {
              resolve(result.data);
            } else {
              reject(new Error(`QRNG API Error: ${result.message}`));
            }
          } catch (e) {
            reject(new Error('Invalid JSON response from QRNG API'));
          }
        });
      });

      req.on('error', reject);
      req.setTimeout(10000, () => {
        req.destroy();
        reject(new Error('QRNG API request timeout'));
      });
      req.end();
    });
  }

  private async startEntropyPooling(): Promise<void> {
    if (this.isPooling) return;
    this.isPooling = true;

    const poolEntropy = async () => {
      try {
        if (this.entropyPool.length < this.poolSize / 4) {
          // Request binary data for pooling
          const endpoint = `/${this.apiKey}/qbck/block/hex?size=50&length=32`;
          const response = await this.makeRequest(endpoint);
          
          if (response.result && Array.isArray(response.result)) {
            response.result.forEach((hex: string) => {
              const buffer = Buffer.from(hex, 'hex');
              this.entropyPool.push(buffer);
            });
          }
        }
      } catch (error) {
        console.warn('Entropy pooling failed:', error);
      }
      
      // Continue pooling every 5 seconds
      setTimeout(poolEntropy, 5000);
    };

    poolEntropy();
  }

  private getPooledBytes(length: number): Buffer | null {
    if (this.entropyPool.length === 0) return null;
    
    const needed = Math.ceil(length / 32);
    if (this.entropyPool.length < needed) return null;
    
    const buffers = this.entropyPool.splice(0, needed);
    const combined = Buffer.concat(buffers);
    return combined.slice(0, length);
  }

  async getRandomBytes(length: number): Promise<Buffer> {
    if (!this.apiKey) {
      throw new Error('QRNG API key not available - true quantum randomness required');
    }

    // Try to use pooled entropy first
    const pooled = this.getPooledBytes(length);
    if (pooled) return pooled;

    const endpoint = `/${this.apiKey}/qbck/block/hex?size=1&length=${length}`;
    const response = await this.makeRequest(endpoint);
    
    if (response.result && response.result[0]) {
      return Buffer.from(response.result[0], 'hex');
    }
    
    throw new Error('No quantum data received from QRNG API - true randomness required');
  }

  async getRandomIntegers(count: number, min: number, max: number): Promise<number[]> {
    if (!this.apiKey) {
      throw new Error('QRNG API key not available - true quantum randomness required');
    }

    const endpoint = `/${this.apiKey}/qbck/block/int?size=${count}&min=${min}&max=${max}`;
    const response = await this.makeRequest(endpoint);
    
    if (response.result && Array.isArray(response.result)) {
      return response.result;
    }
    
    throw new Error('No quantum data received from QRNG API - true randomness required');
  }

  async getRandomFloats(count: number, min: number, max: number): Promise<number[]> {
    if (!this.apiKey) {
      throw new Error('QRNG API key not available - true quantum randomness required');
    }

    const endpoint = `/${this.apiKey}/qbck/block/double?size=${count}&min=${min}&max=${max}`;
    const response = await this.makeRequest(endpoint);
    
    if (response.result && Array.isArray(response.result)) {
      return response.result;
    }
    
    throw new Error('No quantum data received from QRNG API - true randomness required');
  }

  async isAvailable(): Promise<boolean> {
    if (!this.apiKey) return false;
    
    try {
      const endpoint = `/${this.apiKey}/qbck/block/int?size=1&min=0&max=1`;
      await this.makeRequest(endpoint);
      return true;
    } catch (error) {
      return false;
    }
  }

  getEntropyPoolStatus(): { size: number; percentage: number } {
    const size = this.entropyPool.length * 32; // Each buffer is 32 bytes
    const percentage = Math.min(100, (size / this.poolSize) * 100);
    return { size, percentage };
  }
}

export const qrngProvider = new QuantumBlockchainsQRNG();
