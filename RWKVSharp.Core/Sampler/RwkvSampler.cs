using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RWKVSharp.Core.Sampler
{
    public class RwkvSampler : ISampler
    {
        private float _temperature;
        private float _top_p;
        private int _top_k;

        public RwkvSampler(float temperature = 1.0f, float top_p = 0.85f, int top_k = 0)
        {
            _temperature = temperature;
            _top_p = top_p;
            _top_k = top_k;
        }

        public int Sample(IEnumerable<float> logits)
        {
            // Apply softmax
            var maxLogit = logits.Max();
            var expLogits = logits.Select(l => Math.Exp(l - maxLogit));
            var sumExpLogits = expLogits.Sum();
            var probs = expLogits.Select(e => e / sumExpLogits).ToArray();
            // Sort the probabilities 
            var sortedIndices = probs
                .Select((item, index) => new KeyValuePair<double, int>(item, index))
                .OrderByDescending(item => item.Key)
                .Select(item => item.Value)
                .ToArray();
            var sortedProbs = sortedIndices.Select(i => probs[i]).ToArray();

            // Apply top-p and top-k threshold
            double cutoff = 0;
            int index = 0;
            for (; index < sortedProbs.Length; index++)
            {
                if (sortedProbs.Take(index).Sum() >= _top_p)
                {
                    cutoff = sortedProbs[index - 1];
                    break;
                }
            }

            for (int i = 0; i < probs.Length; i++)
            {
                if (probs[i] < cutoff || (_top_k > 0 && i >= _top_k))
                {
                    probs[i] = 0;
                }
            }
            // Apply temperature
            if (_temperature != 1)
            {
                var sum = probs.Sum();
                for (int i = 0; i < probs.Length; i++)
                {
                    probs[i] = Math.Pow(probs[i] / sum, 1.0 / _temperature);
                }
            }

            // Normalize probabilities
            var total = probs.Sum();
            for (int i = 0; i < probs.Length; i++)
            {
                probs[i] /= total;
            }

            // Sample from the distribution
            Random random = new Random();
            double samplePoint = random.NextDouble();
            double currentSum = 0;
            for (int i = 0; i < probs.Length; i++)
            {
                currentSum += probs[i];
                if (currentSum >= samplePoint)
                {
                    return i;
                }
            }
            return 0; // Should not reach here
        }
    }
}
