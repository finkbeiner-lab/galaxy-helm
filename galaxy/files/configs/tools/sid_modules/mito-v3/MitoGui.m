function varargout = MitoGui(varargin)
% MITOGUI MATLAB code for MitoGui.fig
%      MITOGUI, by itself, creates a new MITOGUI or raises the existing
%      singleton*.
%
%      H = MITOGUI returns the handle to a new MITOGUI or the handle to
%      the existing singleton*.
%
%      MITOGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in MITOGUI.M with the given input arguments.
%
%      MITOGUI('Property','Value',...) creates a new MITOGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before MitoGui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to MitoGui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help MitoGui

% Last Modified by GUIDE v2.5 05-Jun-2015 13:38:04

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @MitoGui_OpeningFcn, ...
                   'gui_OutputFcn',  @MitoGui_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before MitoGui is made visible.
function MitoGui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to MitoGui (see VARARGIN)

% Choose default command line output for MitoGui
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes MitoGui wait for user response (see UIRESUME)
% uiwait(handles.main);


% --- Outputs from this function are returned to the command line.
function varargout = MitoGui_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in montageBtn.
function montageBtn_Callback(hObject, eventdata, handles)
% hObject    handle to montageBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
montageFolder = uigetdir(pwd);
if montageFolder
    set(handles.montageTxt, 'String', montageFolder);
end
if exist(montageFolder, 'dir')
    setappdata(handles.main, 'montageFolder', montageFolder);
else
    warndlg('Unable to access the specified montage folder');
end

% --- Executes on button press in maskBtn.
function maskBtn_Callback(hObject, eventdata, handles)
% hObject    handle to maskBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
maskFolder = uigetdir(pwd);
if maskFolder
    set(handles.maskTxt, 'String', maskFolder);
end
if exist(maskFolder, 'dir')
    setappdata(handles.main, 'maskFolder', maskFolder);
else
    warndlg('Unable to access the specified mask folder');
end


% --- Executes on button press in startBtn.
function startBtn_Callback(hObject, eventdata, handles)
% hObject    handle to startBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
warning('off', 'stats:kmeans:EmptyCluster');
set(hObject, 'Enable', 'off');
if isappdata(handles.main, 'montageFolder') && isappdata(handles.main, 'maskFolder')
    % we have the necessary inputs
    try
        montageFolder = getappdata(handles.main, 'montageFolder');
        maskFolder = getappdata(handles.main, 'maskFolder');
        csvFile = [];
        if isappdata(handles.main, 'csvFile')
            csvFile = getappdata(handles.main, 'csvFile');
            if ~exist(csvFile, 'file')
                csvFile = [];
            end
        end
        algos = cellstr(get(handles.algorithmSelect,'String'));
        selected = algos{get(handles.algorithmSelect,'Value')};

        if isequal(selected, 'Multi-Threshold')
            set(handles.txtStatus, 'String', 'Analyzing...');
            pause(1);% just to update gui
            saveSkelFlag = get(handles.chkSaveSkel, 'Value');
            run_mito_analysis_longitudinal(montageFolder, maskFolder, csvFile, saveSkelFlag);
            set(handles.txtStatus, 'String', 'Done');
        elseif isequal(selected, 'FFT Texture')
            set(handles.txtStatus, 'String', 'Analyzing...');
            pause(1);% just to update gui
            run_mito_analysis_longitudinal_glrl(montageFolder, maskFolder);
            set(handles.txtStatus, 'String', 'Done');
        end
    catch e
        set(hObject, 'Enable', 'on');
        set(handles.txtStatus, 'String', '');
        msgString = getReport(e, 'extended', 'hyperlinks', 'off');
        warndlg(sprintf('Encountered error : \n%s\n%s', e.message, msgString));
    end
    set(hObject, 'Enable', 'off');
else
    warndlg('Input folders not specified');
end
set(hObject, 'Enable', 'on');
warning('on', 'stats:kmeans:EmptyCluster');

function maskTxt_Callback(hObject, eventdata, handles)
% hObject    handle to maskTxt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of maskTxt as text
%        str2double(get(hObject,'String')) returns contents of maskTxt as a double


% --- Executes during object creation, after setting all properties.
function maskTxt_CreateFcn(hObject, eventdata, handles)
% hObject    handle to maskTxt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in algorithmSelect.
function algorithmSelect_Callback(hObject, eventdata, handles)
% hObject    handle to algorithmSelect (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

contents = cellstr(get(hObject,'String'));
if isequal(contents{get(hObject,'Value')}, 'Multi-Threshold')
    set(handles.chkSaveSkel, 'Visible', 'on');
else
    set(handles.chkSaveSkel, 'Visible', 'off');
    set(handles.chkSaveSkel, 'Value', 0);
end



% --- Executes during object creation, after setting all properties.
function algorithmSelect_CreateFcn(hObject, eventdata, handles)
% hObject    handle to algorithmSelect (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in chkSaveSkel.
function chkSaveSkel_Callback(hObject, eventdata, handles)
% hObject    handle to chkSaveSkel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of chkSaveSkel


% --- Executes on button press in btnCorrectedTrackCSV.
function btnCorrectedTrackCSV_Callback(hObject, eventdata, handles)
% hObject    handle to btnCorrectedTrackCSV (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[csvFile, p] = uigetfile('*.*');
if csvFile
    set(handles.txtCSV, 'String', csvFile);
end
csvFile = fullfile(p, csvFile);
if exist(csvFile, 'file')
    setappdata(handles.main, 'csvFile', csvFile);
else
    warndlg('Unable to access the specified CSV ffile');
    set(handles.txtCSV, 'String', '');
    setappdata(handles.main, 'csvFile', '');
end


function txtCSV_Callback(hObject, eventdata, handles)
% hObject    handle to txtCSV (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtCSV as text
%        str2double(get(hObject,'String')) returns contents of txtCSV as a double


% --- Executes during object creation, after setting all properties.
function txtCSV_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtCSV (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
