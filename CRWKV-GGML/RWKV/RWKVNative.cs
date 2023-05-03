using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace RWKV
{
    public static partial class RWKVNative
    {
        private const string LIBRARY_NAME = "librwkv";

        /// <summary>
        /// Loads the model from a file and prepares it for inference.
        /// </summary>
        /// <param name="modelFilePath">path to model file in ggml format.</param>
        /// <param name="nThreads">count of threads to use, must be positive.</param>
        /// <returns>Returns NULL on any error. Error messages would be printed to stderr.</returns>
        [LibraryImport(LIBRARY_NAME, EntryPoint = "rwkv_init_from_file", StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr InitFromFile(string modelFilePath, uint nThreads);

        /// <summary>
        /// Evaluates the model for a single token.
        /// </summary>
        /// <param name="ctx"></param>
        /// <param name="token">next token index, in range 0 <= token < n_vocab.</param>
        /// <param name="stateIn">FP32 buffer of size rwkv_get_state_buffer_element_count; or NULL, if this is a first pass.</param>
        /// <param name="stateOut">FP32 buffer of size rwkv_get_state_buffer_element_count. This buffer will be written to.</param>
        /// <param name="logitsOut">FP32 buffer of size rwkv_get_logits_buffer_element_count. This buffer will be written to.</param>
        /// <returns>Returns false on any error. Error messages would be printed to stderr.</returns>
        [LibraryImport(LIBRARY_NAME, EntryPoint = "rwkv_eval")]
        [return: MarshalAs(UnmanagedType.I1)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial bool Eval(IntPtr ctx, int token, IntPtr stateIn, IntPtr stateOut, IntPtr logitsOut);

        /// <summary>
        /// </summary>
        /// <param name="ctx"></param>
        /// <returns>Returns count of FP32 elements in state buffer.</returns>
        [LibraryImport(LIBRARY_NAME, EntryPoint = "rwkv_get_state_buffer_element_count")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial uint GetStateBufferElementCount(IntPtr ctx);

        /// <summary>
        /// </summary>
        /// <param name="ctx"></param>
        /// <returns>Returns count of FP32 elements in logits buffer.</returns>
        [LibraryImport(LIBRARY_NAME, EntryPoint = "rwkv_get_logits_buffer_element_count")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial uint GetLogitsBufferElementCount(IntPtr ctx);

        /// <summary>
        /// Frees all allocated memory and the context.
        /// </summary>
        /// <param name="ctx"></param>
        [LibraryImport(LIBRARY_NAME, EntryPoint = "rwkv_free")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void Free(IntPtr ctx);

        /// <summary>
        /// Quantizes FP32 or FP16 model to one of quantized formats.
        /// Available format names:
        /// - Q4_0
        /// - Q4_1
        /// - Q4_2
        /// - Q5_0
        /// - Q5_1
        /// - Q8_0
        /// </summary>
        /// <param name="modelFilePathIn">path to model file in ggml format, must be either FP32 or FP16.</param>
        /// <param name="modelFilePathOut">quantized model will be written here.</param>
        /// <param name="formatName">must be one of available format names below.</param>
        /// <returns>Returns false on any error. Error messages would be printed to stderr.</returns>
        [LibraryImport(LIBRARY_NAME, EntryPoint = "rwkv_quantize_model_file", StringMarshalling = StringMarshalling.Utf8)]
        [return: MarshalAs(UnmanagedType.I1)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial bool QuantizeModelFile(string modelFilePathIn, string modelFilePathOut, string formatName);

        /// <summary>
        /// </summary>
        /// <returns>Returns system information string.</returns>
        [LibraryImport(LIBRARY_NAME, EntryPoint = "rwkv_get_system_info_string")]
        [return: MarshalAs(UnmanagedType.LPStr)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial string GetSystemInfoString();
    }
}
