using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace RWKVSharp.Core.Tokenizer.RWKV
{
    public class Trie
    {
        private Trie _front;
        private byte? _ch;
        private Trie[] _to;
        private HashSet<(byte[], int)> _values;

        public Trie(Trie front = null, byte? ch = null)
        {
            _ch = ch;
            _to = new Trie[256];
            _values = new HashSet<(byte[], int)>();
            _front = front;
        }

        public Trie Add(byte[] key, int idx = 0, (byte[], int)? val = null)
        {
            if (idx == key.Length)
            {
                if (!val.HasValue)
                    val = (key, idx);
                _values.Add(val.Value);
                return this;
            }
            byte ch = key[idx];
            if (_to[ch] == null)
                _to[ch] = new Trie(front: this, ch: ch);
            return _to[ch].Add(key, idx: idx + 1, val: val);
        }

        public (int index, Trie trie, HashSet<(byte[], int)> values) FindLongest(byte[] key, int idx = 0)
        {
            Trie u = this;
            var ret = default((int, Trie, HashSet<(byte[], int)>));

            while (idx < key.Length && u._to[key[idx]] != null)
            {
                u = u._to[key[idx]];
                idx += 1;
                if (u._values.Count > 0)
                {
                    ret = (idx, u, u._values);
                }
            }
            if (ret == default && u._values.Count > 0)
            {
                ret = (idx, u, u._values);
            }
            return ret;
        }
    }

    public class TrieTokenizer : ITokenizer
    {
        private Trie _root;
        private Dictionary<int, byte[]> _idx2Token;
        private Dictionary<byte[], int> _token2Idx;

        public TrieTokenizer(string fileName)
        {
            _idx2Token = new Dictionary<int, byte[]>();
            _token2Idx = new Dictionary<byte[], int>(new ByteArrayComparer());
            var sorted = new List<byte[]>();
            var lines = File.ReadLines(fileName, Encoding.UTF8);
            foreach (var line in lines)
            {
                if (string.IsNullOrWhiteSpace(line))
                    continue;
                int idx = int.Parse(line[..line.IndexOf(' ')]);
                var f = '"';
                if (Regex.IsMatch(line,$"^{idx}(\\s+)b'") || Regex.IsMatch(line, $"^{idx}(\\s+)'"))
                    f = '\'';
                var q = line.IndexOf(f) + 1;
                var s = line.LastIndexOf(f);
                var value = line[q..s];
                byte[] bytes;
                if (Regex.IsMatch(line, $"^{idx}(\\s+)b'") || Regex.IsMatch(line, $"^{idx}(\\s+)b\""))
                {
                    bytes = StringToByteArray(value.Replace("\\x", ""));
                }
                else
                {
                    string token = Regex.Unescape(value);
                    bytes = Encoding.UTF8.GetBytes(token);
                }
                int bytesLength = int.Parse(line[(line.LastIndexOf(' ') + 1)..]);
                if (bytes.Length != bytesLength)
                {
                    throw new Exception("Byte length mismatch");
                }
                sorted.Add(bytes);
                _idx2Token[idx] = bytes;
                _token2Idx[bytes] = idx;
            }

            _root = new Trie();
            foreach (var tokenAndIdx in _token2Idx)
            {
                _root.Add(tokenAndIdx.Key, val: (tokenAndIdx.Key, tokenAndIdx.Value));
            }
        }

        private byte[] StringToByteArray(String hexString)
        {
            int NumberChars = hexString.Length;
            byte[] bytes = new byte[NumberChars / 2];
            for (int i = 0; i < NumberChars; i += 2)
                bytes[i / 2] = Convert.ToByte(hexString.Substring(i, 2), 16);
            return bytes;
        }

        public List<int> Encode(string text)
        {
            var src = Encoding.UTF8.GetBytes(text);
            int idx = 0;
            var tokens = new List<int>();
            while (idx < src.Length)
            {
                var (nextIndex, _, values) = _root.FindLongest(src, idx);
                idx = nextIndex;
                var (_, token) = values.First();
                tokens.Add(token);
            }
            return tokens;
        }
        public string Decode(IEnumerable<int> tokens)
        {
            return Encoding.UTF8.GetString(tokens.SelectMany(i => _idx2Token[i]).ToArray());
        }
    }

    public class ByteArrayComparer : IEqualityComparer<byte[]>
    {
        public bool Equals(byte[] x, byte[] y)
        {
            if (x.Length != y.Length) return false;
            for (int i = 0; i < x.Length; i++)
                if (x[i] != y[i]) return false;
            return true;
        }
        public int GetHashCode(byte[] obj)
        {
            return obj.GetHashCode();
        }
    }
}
