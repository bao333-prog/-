using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;
using System.Threading.Tasks;

internal sealed class CameraDevice
{
    internal int Index;
    internal string Name;
    internal string Id;
    public override string ToString() { return Name; }
}

internal static class Cameras
{
    [ComImport, Guid("29840822-5B84-11D0-BD3B-00A0C911CE86"),
     InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface ICreateDevEnum
    {
        [PreserveSig]
        int CreateClassEnumerator(ref Guid category, out IEnumMoniker items, int flags);
    }

    [ComImport, Guid("55272A00-42CB-11CE-8135-00AA004BB851"),
     InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IPropertyBag
    {
        [PreserveSig]
        int Read([MarshalAs(UnmanagedType.LPWStr)] string name,
                 [MarshalAs(UnmanagedType.Struct)] out object value, IntPtr log);
    }

    internal static List<CameraDevice> List()
    {
        var devices = new List<CameraDevice>();
        var type = Type.GetTypeFromCLSID(new Guid("62BE5D10-60EB-11D0-BD3B-00A0C911CE86"));
        var factory = (ICreateDevEnum)Activator.CreateInstance(type);
        IEnumMoniker items = null;
        try
        {
            var video = new Guid("860BB310-5D01-11D0-BD3B-00A0C911CE86");
            if (factory.CreateClassEnumerator(ref video, out items, 0) != 0)
                return devices;
            var monikers = new IMoniker[1];
            while (items.Next(1, monikers, IntPtr.Zero) == 0)
            {
                object bag = null;
                try
                {
                    var propertyBag = typeof(IPropertyBag).GUID;
                    monikers[0].BindToStorage(null, null, ref propertyBag, out bag);
                    var properties = (IPropertyBag)bag;
                    object name;
                    object id;
                    properties.Read("FriendlyName", out name, IntPtr.Zero);
                    properties.Read("DevicePath", out id, IntPtr.Zero);
                    devices.Add(new CameraDevice {
                        Index = devices.Count,
                        Name = Convert.ToString(name),
                        Id = id == null ? Convert.ToString(name) + devices.Count : Convert.ToString(id)
                    });
                }
                finally
                {
                    if (bag != null) Marshal.ReleaseComObject(bag);
                    Marshal.ReleaseComObject(monikers[0]);
                }
            }
            return devices;
        }
        finally
        {
            if (items != null) Marshal.ReleaseComObject(items);
            Marshal.ReleaseComObject(factory);
        }
    }
}

internal sealed class CameraStream : IDisposable
{
    private readonly Process process;
    private readonly object gate = new object();
    private byte[] latest;
    private long latestAt;
    internal readonly Stopwatch Age = Stopwatch.StartNew();

    internal CameraStream(string folder, int index)
    {
        var info = RuntimeFiles.Python(folder, "-u camera_worker.py " +
            index.ToString(System.Globalization.CultureInfo.InvariantCulture));
        info.RedirectStandardOutput = true;
        info.RedirectStandardError = true;
        process = Process.Start(info);
        process.ErrorDataReceived += (sender, args) => { };
        process.BeginErrorReadLine();
        var stream = process.StandardOutput.BaseStream;
        Task.Run(() => ReadFrames(stream));
    }

    private void ReadFrames(Stream stream)
    {
        try
        {
            using (var reader = new BinaryReader(stream))
            {
                while (true)
                {
                    int size = reader.ReadInt32();
                    if (size < 1 || size > 16 * 1024 * 1024)
                        break;
                    byte[] frame = reader.ReadBytes(size);
                    if (frame.Length != size)
                        break;
                    lock (gate)
                    {
                        latest = frame;
                        latestAt = Age.ElapsedMilliseconds;
                    }
                }
            }
        }
        catch (IOException) { }
        catch (ObjectDisposedException) { }
        finally
        {
            lock (gate) latest = null;
        }
    }

    internal byte[] FreshFrame()
    {
        lock (gate)
            return Age.ElapsedMilliseconds - latestAt <= 1500 ? latest : null;
    }

    public void Dispose()
    {
        try { if (!process.HasExited) process.Kill(); }
        catch (InvalidOperationException) { }
        finally { process.Dispose(); }
    }
}
