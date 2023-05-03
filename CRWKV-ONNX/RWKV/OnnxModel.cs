using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace RWKV
{
    public enum OnnxModelType
    {
        FP16,
        FP32
    }

    public class OnnxModel : IRWKVModel, IDisposable
    {
        private InferenceSession _inferenceSession;
        private Type _type;
        private int _embed;
        private int _layers;
        private List<string> _input_names;
        private List<string> _output_names;
        private List<NamedOnnxValue> _inputs;
        private OnnxModelType _modelType;

        public OnnxModelType ModelType => _modelType;

        public void Dispose()
        {
            _inferenceSession.Dispose();
        }

        public OnnxModel(string model, int embed, int layers)
        {
            var options = new SessionOptions();
            options.AppendExecutionProvider_CPU();
            options.AppendExecutionProvider_CUDA();
            _inferenceSession = new InferenceSession(model, options);
            _type = _inferenceSession.InputMetadata["instate0"].ElementType;
            _embed = embed;
            _layers = layers;
            _input_names = _inferenceSession.InputMetadata.Select(x => x.Key).ToList();
            _output_names = _inferenceSession.OutputMetadata.Select(x => x.Key).ToList();
            _inputs = new List<NamedOnnxValue>();

            if (_type == typeof(Float16))
            {
                _modelType = OnnxModelType.FP16;
            }
            else if (_type == typeof(float))
            {
                _modelType = OnnxModelType.FP32;
            }
            else
            {
                throw new NotSupportedException();
            }
        }

        public object GetEmptyStates()
        {
            switch (_modelType)
            {
                case OnnxModelType.FP16:
                    {
                        var state = new List<Tensor<Float16>>();
                        for (int i = 0; i < _layers; i++)
                        {
                            state.Add(GDenseTensor<Float16>(0));
                            state.Add(GDenseTensor<Float16>(0));
                            state.Add(GDenseTensor<Float16>(0));
                            state.Add(GDenseTensor<Float16>(0));
                            state.Add(GDenseTensor<Float16>(64512));
                        }
                        return state;
                    }
                case OnnxModelType.FP32:
                    {
                        var state = new List<Tensor<float>>();
                        for (int i = 0; i < _layers; i++)
                        {
                            state.Add(GDenseTensor<float>(0));
                            state.Add(GDenseTensor<float>(0));
                            state.Add(GDenseTensor<float>(0));
                            state.Add(GDenseTensor<float>(0));
                            state.Add(GDenseTensor<float>(float.NegativeInfinity));
                        }
                        return state;
                    };
                default:
                    throw new NotSupportedException();
            }
        }

        public (IEnumerable<float> logits, object state) Forward(int xi, object state)
        {
            switch (_modelType)
            {
                case OnnxModelType.FP16:
                    {
                        var ret = Forward_FP16(xi, (List<Tensor<Float16>>)state);
                        return (ret.logits.Select(x => HalfToSinglePrecision(x)).AsEnumerable(), ret.state);
                    }
                case OnnxModelType.FP32:
                    {
                        var ret = Forward_FP32(xi, (List<Tensor<float>>)state);
                        return (ret.logits.AsEnumerable(), ret.state);
                    }
                default:
                    throw new NotSupportedException();
            }
        }

        private (Tensor<Float16> logits, IList<Tensor<Float16>> state) Forward_FP16(int xi, List<Tensor<Float16>> state)
        {
            _inputs.Clear();
            var input = new DenseTensor<int>(new[] { xi }, new[] { 1 });
            _inputs.Add(NamedOnnxValue.CreateFromTensor(_input_names.First(), input));
            for (int i = 1; i < _input_names.Count; i++)
            {
                _inputs.Add(NamedOnnxValue.CreateFromTensor(_input_names[i], state[i - 1]));
            }
            var data = _inferenceSession.Run(_inputs);
            return (data.First().AsTensor<Float16>(), data.Skip(1).Select(x => x.AsTensor<Float16>()).ToList());
        }

        private (Tensor<float> logits, IList<Tensor<float>> state) Forward_FP32(int xi, IList<Tensor<float>> state)
        {
            _inputs.Clear();
            var input = new DenseTensor<int>(new[] { xi }, new[] { 1 });
            _inputs.Add(NamedOnnxValue.CreateFromTensor(_input_names.First(), input));
            for (int i = 1; i < _input_names.Count; i++)
            {
                _inputs.Add(NamedOnnxValue.CreateFromTensor(_input_names[i], state[i - 1]));
            }
            var data = _inferenceSession.Run(_inputs);
            return (data.First().AsTensor<float>(), data.Skip(1).Select(x => x.AsTensor<float>()).ToList());
        }

        private float HalfToSinglePrecision(ushort half)
        {
            uint sign = (uint)(half >> 15);
            uint exponent = (uint)((half & 0x7C00) >> 10);
            uint mantissa = (uint)(half & 0x03FF);

            uint singleSign = sign << 31;
            uint singleExponent = (exponent + 127 - 15) << 23;
            uint singleMantissa = mantissa << (23 - 10);

            uint singleFloatBits = singleSign | singleExponent | singleMantissa;
            float result = BitConverter.ToSingle(BitConverter.GetBytes(singleFloatBits), 0);

            return result;
        }

        private DenseTensor<T> GDenseTensor<T>(T value)
        {
            var tvalue = new DenseTensor<T>(_embed);
            for (int i2 = 0; i2 < _embed; i2++)
            {
                tvalue[i2] = value;
            }
            return tvalue;
        }
    }
}
