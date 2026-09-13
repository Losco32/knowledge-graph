import { stat } from 'fs/promises';
import { join } from 'path';
import { parseVault } from './parser.js';
import type { Store } from './store.js';
import { Embedder } from './embedder.js';
import { KnowledgeGraph } from './graph.js';

export interface IndexStats {
  nodesIndexed: number;
  nodesSkipped: number;
  edgesIndexed: number;
  communitiesDetected: number;
  stubNodesCreated: number;
}

export class IndexPipeline {
  constructor(
    private store: Store,
    private embedder: Embedder,
  ) {}

  async index(vaultPath: string, resolution = 1.0): Promise<IndexStats> {
    const stats: IndexStats = {
      nodesIndexed: 0,
      nodesSkipped: 0,
      edgesIndexed: 0,
      communitiesDetected: 0,
      stubNodesCreated: 0,
    };

    const { nodes, edges, stubIds } = await parseVault(vaultPath);
    const previousPaths = this.store.getAllSyncPaths();

    // Detect deleted files
    const currentPaths = new Set(nodes.map(n => n.id));
    for (const oldPath of previousPaths) {
      if (!currentPaths.has(oldPath)) {
        this.store.deleteNode(oldPath);
      }
    }

    // Index nodes (incremental)
    for (const node of nodes) {
      const fileStat = await stat(join(vaultPath, node.id));
      const mtime = fileStat.mtimeMs;
      const prevMtime = this.store.getSyncMtime(node.id);

      if (prevMtime !== undefined && prevMtime >= mtime) {
        stats.nodesSkipped++;
        continue;
      }

      this.store.upsertNode(node);

      // Chunk the body into overlapping ~200-token windows and embed each
      // chunk separately, so long notes are retrievable by their relevant
      // passage rather than diluted into one whole-note vector.
      const chunks = await this.embedder.embedChunks(node.content);
      this.store.upsertEmbeddings(node.id, chunks);

      // Re-index edges from this node
      this.store.deleteAllEdgesFrom(node.id);
      for (const edge of edges.filter(e => e.sourceId === node.id)) {
        this.store.insertEdge(edge);
        stats.edgesIndexed++;
      }

      this.store.upsertSync(node.id, mtime);
      stats.nodesIndexed++;
    }

    // Create stub nodes
    for (const stubId of stubIds) {
      if (!this.store.getNode(stubId)) {
        this.store.upsertNode({
          id: stubId,
          title: stubId.replace('_stub/', '').replace('.md', ''),
          content: '',
          frontmatter: { _stub: true },
        });
        stats.stubNodesCreated++;
      }
    }

    // If any nodes were indexed, re-run community detection
    if (stats.nodesIndexed > 0 || stats.stubNodesCreated > 0) {
      const kg = KnowledgeGraph.fromStore(this.store);
      const communities = kg.detectCommunities(resolution);
      this.store.clearCommunities();
      for (const c of communities) {
        this.store.upsertCommunity(c);
      }
      stats.communitiesDetected = communities.length;
    }

    return stats;
  }
}
