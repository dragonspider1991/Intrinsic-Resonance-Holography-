import time
from irh import HyperGraph
from irh.spectral_dimension import SpectralDimension
from irh.grand_audit import grand_audit

def main():
    print('=== IRH v9.2 Benchmark Suite ===')
    print()

    # Benchmark N=64
    start = time.time()
    G = HyperGraph(N=64, seed=42)
    print(f'Create N=64 graph: {time.time()-start:.3f}s')

    start = time.time()
    ds = SpectralDimension(G)
    print(f'Spectral dimension: {time.time()-start:.3f}s (d_s={ds.value:.2f})')

    # Benchmark N=256
    start = time.time()
    G = HyperGraph(N=256, seed=42)
    print(f'Create N=256 graph: {time.time()-start:.3f}s')

    start = time.time()
    ds = SpectralDimension(G)
    print(f'Spectral dimension: {time.time()-start:.3f}s (d_s={ds.value:.2f})')

    # Grand audit on small graph
    G = HyperGraph(N=32, seed=42)
    start = time.time()
    report = grand_audit(G)
    print(f'Grand audit N=32: {time.time()-start:.3f}s')
    print(f'  Pass rate: {report.pass_count}/{report.total_checks}')

if __name__ == '__main__':
    main()
