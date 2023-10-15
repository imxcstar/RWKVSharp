using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CLLM.Core.Sampler
{
    public class NPSampler : ISampler
    {
        private float _temp;
        private float _top_p_usual;

        public NPSampler(float temp = 1.0f, float top_p_usual = 0.8f)
        {
            _temp = temp;
            _top_p_usual = top_p_usual;
        }

        public int Sample(IEnumerable<float> logits)
        {
            // 使用MathNet.Numerics库创建一个向量。
            var vector = Vector<float>.Build.DenseOfEnumerable(logits);

            // 应用softmax功能。
            float maxElem = vector.Maximum();
            vector = vector.Subtract(maxElem);
            vector = vector.PointwiseExp();
            float sumEXP = vector.Sum();
            vector = vector.Divide(sumEXP);

            // 计算累积概率。
            var sorted_probs = vector.OrderByDescending(x => x).ToArray();
            var cumulative_probs = new float[sorted_probs.Length];
            for (int i = 0; i < sorted_probs.Length; i++)
            {
                cumulative_probs[i] = (i == 0) ? sorted_probs[i] : cumulative_probs[i - 1] + sorted_probs[i];
            }

            // 查找截止点。
            float cutoff = 0;
            for (int i = 0; i < sorted_probs.Length; i++)
            {
                if (cumulative_probs[i] > _top_p_usual)
                {
                    cutoff = sorted_probs[i];
                    break;
                }
            }
            int[] indices_below_cutoff = vector.Select((item, index) => item < cutoff ? index : -1).Where(index => index >= 0).ToArray();

            if (indices_below_cutoff.Length > 0)
            {
                foreach (int index in indices_below_cutoff)
                {
                    vector[index] = 0;
                }

                vector /= vector.Sum();
            }

            if (_temp != 1.0)
            {
                vector = vector.PointwisePower(1.0f / _temp);
                vector /= vector.Sum();
            }

            // 使用Categorical分布，根据调整后的概率分布选择一个随机整数。
            var categorical = new Categorical(vector.Select(x => Convert.ToDouble(x)).ToArray());
            int mout = categorical.Sample();

            return mout;
        }
    }
}
