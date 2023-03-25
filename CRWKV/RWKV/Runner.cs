using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp;

namespace RWKV
{
    public class Runner
    {
        private Tokenizer _tokenizer;
        private InferenceSession _inferenceSession;

        private int _ctx_len = 0;
        private int _n_layer = 0;
        private int _n_embd = 0;

        private string _model;

        public Runner(string model, int ctx_len, int n_layer, int n_embd)
        {
            _model = model;
            _ctx_len = ctx_len;
            _n_layer = n_layer;
            _n_embd = n_embd;
        }

        public int Init()
        {
            TensorAllocator.InitDevices(ProcessorTypeEnums.CPU, new int[] { 0 });

            var path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Model");
            _tokenizer = new Tokenizer(Path.Combine(path, "20B_tokenizer.json"));

            _inferenceSession = new InferenceSession(Path.Combine(path, _model));
            return _ctx_len;
        }

        public void Run(string value, Action<string?> callBack)
        {
            var ctx = _tokenizer.Encoder.Encode(value);

            Tensor<float> xx_att = new DenseTensor<float>(new[] { _n_layer, _n_embd });
            Tensor<float> aa_att = new DenseTensor<float>(new[] { _n_layer, _n_embd });
            Tensor<float> bb_att = new DenseTensor<float>(new[] { _n_layer, _n_embd });
            Tensor<float> pp_att = new DenseTensor<float>(new[] { _n_layer, _n_embd });
            Tensor<float> xx_ffn = new DenseTensor<float>(new[] { _n_layer, _n_embd });
            Tensor<int> input = new DenseTensor<int>(new[] { _ctx_len });

            Queue<int> xutput = new Queue<int>(ctx);
            var size = xutput.Count;
            for (int i = 0; i < size + 200; i++)
            {
                if (xutput.Count > 0)
                {
                    input[_ctx_len - 1] = xutput.Dequeue();
                }
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("idx", input),
                    NamedOnnxValue.CreateFromTensor("xx_att", xx_att),
                    NamedOnnxValue.CreateFromTensor("aa_att", aa_att),
                    NamedOnnxValue.CreateFromTensor("bb_att", bb_att),
                    NamedOnnxValue.CreateFromTensor("pp_att", pp_att),
                    NamedOnnxValue.CreateFromTensor("xx_ffn", xx_ffn),
                };
                var ret = _inferenceSession.Run(inputs).ToArray();
                var f = ret[0].AsTensor<float>();
                xx_att = ret[1].AsTensor<float>();
                aa_att = ret[2].AsTensor<float>();
                bb_att = ret[3].AsTensor<float>();
                pp_att = ret[4].AsTensor<float>();
                xx_ffn = ret[5].AsTensor<float>();


                if (xutput.Count == 0)
                {
                    var ac = sl(f);

                    input[_ctx_len - 1] = ac;

                    var ch = _tokenizer.Encoder.Decode(new[] { ac });
                    callBack?.Invoke(ch);
                }
            }
        }

        private int sl(Tensor<float> out1)
        {
            var graph = new ComputeGraphTensor(new WeightTensorFactory(), 0, false);

            var v = out1.AsEnumerable().ToArray();
            var w = new WeightTensor(new long[2] { 1, v.Length }, 0, 0);
            w.SetWeightArray(v);

            var w2 = graph.Softmax(w);

            var data = graph.TopPSampleIndice(w2, new List<List<int>> { new List<int>() });
            var ret = data.ToWeightArray();

            return (int)ret[0];
        }
    }
}
