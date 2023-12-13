using CLLM.Core.Interfaces;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace RWKV
{
    public enum OnnxModelType
    {
        FP16,
        FP32
    }

    public class OnnxModel : IModel, IDisposable
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
            try
            {
                options.AppendExecutionProvider_CUDA();
            }
            catch (Exception)
            {
            }
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
                        var state2 = new List<Tensor<Float16>>();
                        for (int i = 0; i < _layers; i++)
                        {
                            var tstate= new DenseTensor<Float16>(new int[] { _embed });
                            var tstate1 = new DenseTensor<Float16>(new int[] { _embed });
                            var tstate2 = new DenseTensor<Float16>(new int[] { 40, 64, 64 });
                            tstate.Fill(new Float16(0));
                            tstate1.Fill(new Float16(0));
                            tstate2.Fill(new Float16(0));
                            state.Add(tstate);
                            state.Add(tstate1);
                            state2.Add(tstate2);
                        }
                        return (state, state2);
                    }
                case OnnxModelType.FP32:
                    {
                        var state = new List<Tensor<float>>();
                        var state2 = new List<Tensor<float>>();
                        for (int i = 0; i < _layers; i++)
                        {
                            var tstate = new DenseTensor<float>(new int[] { _embed });
                            var tstate1 = new DenseTensor<float>(new int[] { _embed });
                            var tstate2 = new DenseTensor<float>(new int[] { 40, 64, 64 });
                            tstate.Fill(0.01f);
                            tstate1.Fill(0.01f);
                            tstate2.Fill(0.01f);
                            state.Add(tstate);
                            state.Add(tstate1);
                            state2.Add(tstate2);
                        }
                        return (state, state2);
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
                        var ret = Forward_FP16(xi, ((List<Tensor<Float16>>, List<Tensor<Float16>>))state);
                        return (ret.logits.Select(x => x.ToFloat()).AsEnumerable(), ret.state);
                    }
                case OnnxModelType.FP32:
                    {
                        var ret = Forward_FP32(xi, ((List<Tensor<float>>, List<Tensor<float>>))state);
                        return (ret.logits.AsEnumerable(), ret.state);
                    }
                default:
                    throw new NotSupportedException();
            }
        }

        private (Tensor<Float16> logits, (List<Tensor<Float16>>, List<Tensor<Float16>>) state) Forward_FP16(int xi, (List<Tensor<Float16>>, List<Tensor<Float16>>) state)
        {
            _inputs.Clear();
            var input = new DenseTensor<int>(new[] { xi }, new[] { 1 });
            _inputs.Add(NamedOnnxValue.CreateFromTensor(_input_names.First(), input));
            for (int i = 1; i < _input_names.Count; i++)
            {
                if (_input_names[i].Contains("wkv"))
                {
                    _inputs.Add(NamedOnnxValue.CreateFromTensor(_input_names[i], state.Item2[i - state.Item1.Count]));
                }
                else
                {
                    _inputs.Add(NamedOnnxValue.CreateFromTensor(_input_names[i], state.Item1[i - 1]));
                }
            }
            var data = _inferenceSession.Run(_inputs);
            return (data.First().AsTensor<Float16>(), (data.Skip(1).Take(state.Item1.Count + 1).Select(x => x.AsTensor<Float16>()).ToList(), data.Skip(state.Item1.Count + 1).Select(x => x.AsTensor<Float16>()).ToList()));
        }

        private (Tensor<float> logits, (List<Tensor<float>>, List<Tensor<float>>) state) Forward_FP32(int xi, (List<Tensor<float>>, List<Tensor<float>>) state)
        {
            _inputs.Clear();
            var input = new DenseTensor<int>(new[] { xi }, new[] { 1 });
            _inputs.Add(NamedOnnxValue.CreateFromTensor(_input_names.First(), input));
            for (int i = 1; i < _input_names.Count; i++)
            {
                if (_input_names[i].Contains("wkv"))
                {
                    _inputs.Add(NamedOnnxValue.CreateFromTensor(_input_names[i], state.Item2[i - state.Item1.Count - 1]));
                }
                else
                {
                    _inputs.Add(NamedOnnxValue.CreateFromTensor(_input_names[i], state.Item1[i - 1]));
                }
            }
            var data = _inferenceSession.Run(_inputs);
            return (data.First().AsTensor<float>(), (data.Skip(1).Take(state.Item1.Count + 1).Select(x => x.AsTensor<float>()).ToList(), data.Skip(state.Item1.Count + 1).Select(x => x.AsTensor<float>()).ToList()));
        }

        public object GetStates(int[] tokens)
        {
            throw new NotImplementedException();
        }
    }
}
