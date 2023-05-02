using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using Seq2SeqSharp.Tools;

namespace RWKV
{
    public class Runner
    {
        private readonly OnnxModel _onnxModel;
        private readonly BPEncoder _encoder;
        private object _state;

        public Runner(OnnxModel onnxModel, BPEncoder encoder)
        {
            _onnxModel = onnxModel;
            _encoder = encoder;
            _state = _onnxModel.GetEmptyStates();
        }

        public void Run(string value, Action<string?> callBack)
        {
            var xutput = new Queue<int>(_encoder.Encode(value));
            var size = xutput.Count;
            var input = 0;
            var strEnc = new List<int>();
            for (int i = 0; i < size + 1024; i++)
            {
                if (xutput.Count > 0)
                {
                    input = xutput.Dequeue();
                }
                var logits_state = _onnxModel.Forward(input, _state);
                _state = logits_state.state;

                if (xutput.Count == 0)
                {
                    var ac = TopPSample(logits_state.logits);
                    input = ac;
                    strEnc.Add(ac);
                    var str = _encoder.Decode(strEnc);
                    if (str == "<|endoftext|>")
                        break;
                    if (!str.Contains("\ufffd"))
                    {
                        callBack?.Invoke(str);
                        strEnc.Clear();
                    }
                }
            }
        }

        private int TopPSample(IEnumerable<float> ozut)
        {
            var graph = new ComputeGraphTensor(new WeightTensorFactory(), 0, false);

            var v = ozut.ToArray();
            var w = new WeightTensor(new long[2] { 1, v.Length }, 0, 0);
            w.SetWeightArray(v);

            var w2 = graph.Softmax(w);

            var data = graph.TopPSampleIndice(w2, new List<List<int>> { new List<int>() });
            var ret = data.ToWeightArray();

            return (int)ret[0];
        }

        public int NPSample(IEnumerable<float> ozut, float temp = 1.0f, float top_p_usual = 0.8f)
        {
            // 使用MathNet.Numerics库创建一个向量。
            var vector = Vector<float>.Build.DenseOfEnumerable(ozut);

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
                if (cumulative_probs[i] > top_p_usual)
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

            if (temp != 1.0)
            {
                vector = vector.PointwisePower(1.0f / temp);
                vector /= vector.Sum();
            }

            // 使用Categorical分布，根据调整后的概率分布选择一个随机整数。
            var categorical = new Categorical(vector.Select(x => Convert.ToDouble(x)).ToArray());
            int mout = categorical.Sample();

            return mout;
        }
    }
}
