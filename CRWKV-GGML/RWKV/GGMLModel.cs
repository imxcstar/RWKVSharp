using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace RWKV
{
    public class GGMLModel : IRWKVModel, IDisposable
    {
        private string _modelPath;
        private IntPtr _model;
        private int _stateCount;
        private int _logitsCount;

        public GGMLModel(string model)
        {
            _modelPath = model;
            _model = RWKVNative.InitFromFile(model, (uint)(Math.Max(1, Environment.ProcessorCount / 2)));
            _stateCount = (int)RWKVNative.GetStateBufferElementCount(_model);
            _logitsCount = (int)RWKVNative.GetLogitsBufferElementCount(_model);
        }

        public void Dispose()
        {
            RWKVNative.Free(_model);
        }

        public object GetEmptyStates()
        {
            return IntPtr.Zero;
        }

        public (IEnumerable<float> logits, object state) Forward(int token, object state)
        {
            var outStateBuffer = Marshal.AllocHGlobal(_stateCount * sizeof(float));
            var outLogitsBuffer = Marshal.AllocHGlobal(_logitsCount * sizeof(float));

            if (!RWKVNative.Eval(_model, token, (IntPtr)state, outStateBuffer, outLogitsBuffer))
                throw new Exception();

            var logits = IntPtrToFloatArray(outLogitsBuffer, _logitsCount);
            return (logits, outStateBuffer);
        }

        private float[] IntPtrToFloatArray(IntPtr floatPtr, int size)
        {
            float[] floatArray = new float[size];
            Marshal.Copy(floatPtr, floatArray, 0, size);
            return floatArray;
        }
    }
}
