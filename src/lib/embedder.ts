import { pipeline, type FeatureExtractionPipeline } from '@huggingface/transformers';

const MODEL = 'Xenova/all-MiniLM-L6-v2';

// Chunk size/overlap are measured in the model's own tokens (not words/chars),
// since chunks are built by slicing token ids and decoding back to text.
const CHUNK_SIZE_TOKENS = 200;
const CHUNK_OVERLAP_TOKENS = 40;

export interface TextChunk {
  index: number;
  text: string;
}

export interface EmbeddedChunk extends TextChunk {
  embedding: Float32Array;
}

export class Embedder {
  private extractor: FeatureExtractionPipeline | null = null;

  async init(): Promise<void> {
    this.extractor = await pipeline('feature-extraction', MODEL, {
      dtype: 'q8',
    }) as FeatureExtractionPipeline;
  }

  async embed(text: string): Promise<Float32Array> {
    if (!this.extractor) throw new Error('Embedder not initialized. Call init() first.');
    const output = await this.extractor(text, {
      pooling: 'mean',
      normalize: true,
    });
    return new Float32Array(output.tolist()[0] as number[]);
  }

  /**
   * Splits text into overlapping chunks of ~chunkSize model tokens. Windows
   * are built on whitespace-delimited words sliced directly out of the
   * original text (so chunk text is an exact substring — original casing,
   * accents, punctuation, spacing), sized by a tokens-per-word ratio measured
   * with the real tokenizer. We deliberately avoid encode()+decode() to build
   * the chunk text itself: this tokenizer lowercases and strips accents (e.g.
   * "perché" -> "perche"), which would corrupt excerpts for non-English notes
   * and can't be reliably mapped back to the original substring afterwards.
   */
  chunkText(
    text: string,
    chunkSize = CHUNK_SIZE_TOKENS,
    overlap = CHUNK_OVERLAP_TOKENS,
  ): TextChunk[] {
    if (!this.extractor) throw new Error('Embedder not initialized. Call init() first.');
    const trimmed = text.trim();
    if (!trimmed) return [];

    const words = [...trimmed.matchAll(/\S+/g)];
    if (words.length === 0) return [];

    const tokenizer = this.extractor.tokenizer;
    const totalTokens = (tokenizer.encode(trimmed, { add_special_tokens: false }) as number[]).length;
    const tokensPerWord = totalTokens / words.length || 1;

    const wordChunkSize = Math.max(Math.round(chunkSize / tokensPerWord), 1);
    const wordOverlap = Math.min(Math.round(overlap / tokensPerWord), wordChunkSize - 1);
    const step = Math.max(wordChunkSize - wordOverlap, 1);

    const chunks: TextChunk[] = [];
    for (let start = 0, index = 0; start < words.length; start += step, index++) {
      const end = Math.min(start + wordChunkSize, words.length);
      const startOffset = words[start].index!;
      const lastWord = words[end - 1];
      const endOffset = lastWord.index! + lastWord[0].length;
      chunks.push({ index, text: trimmed.slice(startOffset, endOffset) });
      if (end >= words.length) break;
    }
    return chunks;
  }

  /** Chunks text and embeds each chunk, preserving chunk index/text alongside its vector. */
  async embedChunks(
    text: string,
    chunkSize = CHUNK_SIZE_TOKENS,
    overlap = CHUNK_OVERLAP_TOKENS,
  ): Promise<EmbeddedChunk[]> {
    const chunks = this.chunkText(text, chunkSize, overlap);
    const embedded: EmbeddedChunk[] = [];
    for (const chunk of chunks) {
      const embedding = await this.embed(chunk.text);
      embedded.push({ ...chunk, embedding });
    }
    return embedded;
  }

  async dispose(): Promise<void> {
    if (this.extractor) {
      await this.extractor.dispose();
      this.extractor = null;
    }
  }

  static buildEmbeddingText(
    title: string,
    tags: string[],
    content: string,
  ): string {
    const firstParagraph = content.split(/\n\n+/)[0] ?? '';
    const parts = [title];
    if (tags.length > 0) {
      parts.push(tags.join(', '));
    }
    if (firstParagraph) {
      parts.push(firstParagraph);
    }
    return parts.join('\n');
  }
}
