using CLLM.Core.Interfaces;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace RWKV
{
    public class GGMLModel : IModel, IDisposable
    {
        private string _modelPath;
        private IntPtr _model;
        private int _stateCount;
        private int _logitsCount;

        public GGMLModel(string model, uint? n_threads = null, uint? n_gpu_layers = null)
        {
            _modelPath = model;
            _model = RwkvCppNative.rwkv_init_from_file(model, n_threads ?? (uint)(Math.Max(1, Environment.ProcessorCount - 4)), n_gpu_layers ?? 0);
            _stateCount = (int)RwkvCppNative.rwkv_get_state_len(_model);
            _logitsCount = (int)RwkvCppNative.rwkv_get_logits_len(_model);
        }

        public void Dispose()
        {
            RwkvCppNative.rwkv_free(_model);
        }

        public object GetEmptyStates()
        {
            return IntPtr.Zero;
        }

        public (IEnumerable<float> logits, object state) Forward(int token, object state)
        {
            var outLogitsBuffer = Marshal.AllocHGlobal(_logitsCount * sizeof(float));
            var outStateBuffer = Marshal.AllocHGlobal(_stateCount * sizeof(float));
            IntPtr inStateBuffer;
            var isFreeInState = false;
            if (state is IntPtr istate)
                inStateBuffer = istate;
            else if (state is float[] fstate)
            {
                isFreeInState = true;
                inStateBuffer = FloatArrayToIntPtr(fstate, fstate.Length);
            }
            else
                throw new NotSupportedException($"not support state type \"{state.GetType()}\"");

            if (!RwkvCppNative.rwkv_eval(_model, (uint)token, inStateBuffer, outStateBuffer, outLogitsBuffer))
                throw new Exception();

            var outLogits = IntPtrToFloatArray(outLogitsBuffer, _logitsCount);
            var outState = IntPtrToFloatArray(outStateBuffer, _stateCount);

            if (isFreeInState)
                Marshal.FreeHGlobal(inStateBuffer);
            Marshal.FreeHGlobal(outStateBuffer);
            Marshal.FreeHGlobal(outLogitsBuffer);

            return (outLogits, outState);
        }

        private float[] IntPtrToFloatArray(IntPtr floatPtr, int size)
        {
            float[] floatArray = new float[size];
            Marshal.Copy(floatPtr, floatArray, 0, size);
            return floatArray;
        }

        private static IntPtr FloatArrayToIntPtr(float[] floatArray, int size)
        {
            IntPtr floatPtr = Marshal.AllocHGlobal(size * sizeof(float));
            Marshal.Copy(floatArray, 0, floatPtr, size);
            return floatPtr;
        }

        public object GetStates(int[] tokens)
        {
            var outLogitsBuffer = Marshal.AllocHGlobal(_logitsCount * sizeof(float));
            var outStateBuffer = Marshal.AllocHGlobal(_stateCount * sizeof(float));

            if (!RwkvCppNative.rwkv_eval_sequence_in_chunks(_model, tokens.Select(x => (uint)x).ToArray(), (ulong)tokens.Count(), 3072, IntPtr.Zero, outStateBuffer, outLogitsBuffer))
                throw new Exception();

            var outLogits = IntPtrToFloatArray(outLogitsBuffer, _logitsCount);
            var outState = IntPtrToFloatArray(outStateBuffer, _stateCount);

            Marshal.FreeHGlobal(outStateBuffer);
            Marshal.FreeHGlobal(outLogitsBuffer);

            return outState;
        }
    }
}
