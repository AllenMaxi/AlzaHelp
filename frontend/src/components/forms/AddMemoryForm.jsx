import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Plus, Upload, X, MapPin, Calendar, Users, Loader2, Mic, MicOff } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';
import { memoriesApi, uploadApi } from '@/services/api';
import { toast } from 'sonner';

const categoryOptions = [
  { value: 'milestone', label: 'Milestone (Wedding, Birth, Graduation)' },
  { value: 'family', label: 'Family Moment' },
  { value: 'travel', label: 'Travel / Vacation' },
  { value: 'celebration', label: 'Celebration / Holiday' },
  { value: 'other', label: 'Other' },
];

// Voice-to-text hook
const useVoiceToText = () => {
  const [isListening, setIsListening] = useState(false);
  const [isSupported, setIsSupported] = useState(false);
  const recognitionRef = useRef(null);
  const onResultRef = useRef(null);

  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    setIsSupported(!!SpeechRecognition);

    if (SpeechRecognition) {
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = false;
      recognitionRef.current.lang = 'en-US';

      recognitionRef.current.onresult = (event) => {
        let transcript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
          if (event.results[i].isFinal) {
            transcript += event.results[i][0].transcript + ' ';
          }
        }
        if (transcript && onResultRef.current) {
          onResultRef.current(transcript);
        }
      };

      recognitionRef.current.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        setIsListening(false);
        if (event.error === 'not-allowed') {
          toast.error('Please allow microphone access to use voice input');
        }
      };

      recognitionRef.current.onend = () => {
        setIsListening(false);
      };
    }

    return () => {
      if (recognitionRef.current) {
        try {
          recognitionRef.current.stop();
        } catch (e) {}
      }
    };
  }, []);

  const startListening = useCallback((onResult) => {
    if (recognitionRef.current && !isListening) {
      onResultRef.current = onResult;
      recognitionRef.current.start();
      setIsListening(true);
    }
  }, [isListening]);

  const stopListening = useCallback(() => {
    if (recognitionRef.current && isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    }
  }, [isListening]);

  return { isListening, isSupported, startListening, stopListening };
};

export const AddMemoryForm = ({ familyMembers = [], onSuccess, onClose }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [uploadingPhoto, setUploadingPhoto] = useState(false);
  const [personInput, setPersonInput] = useState('');
  
  const { isListening, isSupported, startListening, stopListening } = useVoiceToText();
  
  const [formData, setFormData] = useState({
    title: '',
    date: '',
    year: new Date().getFullYear(),
    location: '',
    description: '',
    people: [],
    photos: [],
    category: '',
  });

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleYearChange = (e) => {
    const value = parseInt(e.target.value) || new Date().getFullYear();
    setFormData(prev => ({ ...prev, year: value }));
  };

  const handleCategoryChange = (value) => {
    setFormData(prev => ({ ...prev, category: value }));
  };

  const handlePhotoUpload = async (e) => {
    const files = Array.from(e.target.files);
    if (files.length === 0) return;
    
    setUploadingPhoto(true);
    try {
      const urls = await Promise.all(files.map(file => uploadApi.uploadFile(file)));
      setFormData(prev => ({ ...prev, photos: [...prev.photos, ...urls] }));
      toast.success('Photo uploaded successfully');
    } catch (error) {
      toast.error('Failed to upload photo');
    } finally {
      setUploadingPhoto(false);
    }
  };

  const removePhoto = (index) => {
    setFormData(prev => ({
      ...prev,
      photos: prev.photos.filter((_, i) => i !== index)
    }));
  };

  const addPerson = (name) => {
    if (name && !formData.people.includes(name)) {
      setFormData(prev => ({ ...prev, people: [...prev.people, name] }));
    }
    setPersonInput('');
  };

  const removePerson = (name) => {
    setFormData(prev => ({
      ...prev,
      people: prev.people.filter(p => p !== name)
    }));
  };

  // Voice input handler
  const handleVoiceInput = () => {
    if (isListening) {
      stopListening();
    } else {
      startListening((transcript) => {
        setFormData(prev => ({
          ...prev,
          description: prev.description + transcript
        }));
      });
      toast.info('Listening... speak now to add to your story');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!formData.title || !formData.date || !formData.category || !formData.description) {
      toast.error('Please fill in the required fields');
      return;
    }
    
    setLoading(true);
    try {
      await memoriesApi.create(formData);
      toast.success('Memory saved successfully!');
      setFormData({
        title: '',
        date: '',
        year: new Date().getFullYear(),
        location: '',
        description: '',
        people: [],
        photos: [],
        category: '',
      });
      setIsOpen(false);
      if (onSuccess) onSuccess();
      if (onClose) onClose();
    } catch (error) {
      toast.error('Failed to save memory');
    } finally {
      setLoading(false);
    }
  };

  const dialogContent = (
    <form onSubmit={handleSubmit} className="space-y-6 mt-4">
      {/* Title */}
      <div>
        <Label htmlFor="title" className="text-lg font-semibold">Memory Title *</Label>
        <Input
          id="title"
          name="title"
          value={formData.title}
          onChange={handleInputChange}
          placeholder="e.g., Our Wedding Day, Family Vacation to Paris"
          className="mt-2 h-14 text-lg"
          required
        />
      </div>

      {/* Date and Year */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <Label htmlFor="date" className="text-lg font-semibold flex items-center gap-2">
            <Calendar className="h-5 w-5" /> Date *
          </Label>
          <Input
            id="date"
            name="date"
            value={formData.date}
            onChange={handleInputChange}
            placeholder="e.g., June 15, 1972"
            className="mt-2 h-14 text-lg"
            required
          />
        </div>
        <div>
          <Label htmlFor="year" className="text-lg font-semibold">Year *</Label>
          <Input
            id="year"
            name="year"
            type="number"
            value={formData.year}
            onChange={handleYearChange}
            min="1900"
            max={new Date().getFullYear()}
            className="mt-2 h-14 text-lg"
            required
          />
        </div>
      </div>

      {/* Category */}
      <div>
        <Label className="text-lg font-semibold">Category *</Label>
        <Select value={formData.category} onValueChange={handleCategoryChange}>
          <SelectTrigger className="mt-2 h-14 text-lg">
            <SelectValue placeholder="Select memory type" />
          </SelectTrigger>
          <SelectContent>
            {categoryOptions.map(opt => (
              <SelectItem key={opt.value} value={opt.value} className="text-lg py-3">
                {opt.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Location */}
      <div>
        <Label htmlFor="location" className="text-lg font-semibold flex items-center gap-2">
          <MapPin className="h-5 w-5" /> Location
        </Label>
        <Input
          id="location"
          name="location"
          value={formData.location}
          onChange={handleInputChange}
          placeholder="e.g., St. Mary's Church, Paris, France"
          className="mt-2 h-14 text-lg"
        />
      </div>

      {/* People */}
      <div>
        <Label className="text-lg font-semibold flex items-center gap-2">
          <Users className="h-5 w-5" /> People in this Memory
        </Label>
        <div className="mt-2 flex gap-2">
          <Input
            value={personInput}
            onChange={(e) => setPersonInput(e.target.value)}
            placeholder="Add a person's name"
            className="h-14 text-lg flex-1"
            onKeyPress={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                addPerson(personInput);
              }
            }}
          />
          <Button
            type="button"
            variant="outline"
            className="h-14 px-6"
            onClick={() => addPerson(personInput)}
          >
            Add
          </Button>
        </div>
        
        {/* Quick add from family members */}
        {familyMembers.length > 0 && (
          <div className="mt-3">
            <p className="text-sm text-muted-foreground mb-2">Quick add from family:</p>
            <div className="flex flex-wrap gap-2">
              {familyMembers.slice(0, 6).map(member => (
                <Button
                  key={member.id}
                  type="button"
                  variant="ghost"
                  size="sm"
                  className={`${formData.people.includes(member.name) ? 'bg-primary/20' : ''}`}
                  onClick={() => {
                    if (formData.people.includes(member.name)) {
                      removePerson(member.name);
                    } else {
                      addPerson(member.name);
                    }
                  }}
                >
                  {member.name}
                </Button>
              ))}
            </div>
          </div>
        )}
        
        {formData.people.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-3">
            {formData.people.map(person => (
              <Badge
                key={person}
                variant="secondary"
                className="text-base py-1 px-3 gap-2"
              >
                {person}
                <button
                  type="button"
                  onClick={() => removePerson(person)}
                  className="hover:text-destructive"
                >
                  <X className="h-4 w-4" />
                </button>
              </Badge>
            ))}
          </div>
        )}
      </div>

      {/* Description with Voice Input */}
      <div>
        <Label htmlFor="description" className="text-lg font-semibold flex items-center gap-2">
          Tell the Story *
          {isSupported && (
            <span className="text-sm font-normal text-muted-foreground ml-2">
              (tap mic to speak)
            </span>
          )}
        </Label>
        <div className="mt-2 relative">
          <textarea
            id="description"
            name="description"
            value={formData.description}
            onChange={handleInputChange}
            placeholder="Describe this memory in detail. What happened? How did you feel? What made it special? You can also tap the microphone to speak your story."
            className="w-full min-h-[150px] rounded-xl border-2 border-input bg-background px-4 py-3 text-lg focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20 resize-none pr-16"
            required
          />
          {isSupported && (
            <Button
              type="button"
              variant={isListening ? "destructive" : "outline"}
              size="icon"
              onClick={handleVoiceInput}
              className="absolute bottom-3 right-3 h-12 w-12"
              title={isListening ? 'Stop recording' : 'Start voice input'}
            >
              {isListening ? (
                <div className="relative">
                  <MicOff className="h-6 w-6" />
                  <span className="absolute -top-1 -right-1 flex h-3 w-3">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-destructive-foreground opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-3 w-3 bg-destructive-foreground"></span>
                  </span>
                </div>
              ) : (
                <Mic className="h-6 w-6" />
              )}
            </Button>
          )}
        </div>
        {isListening && (
          <p className="text-sm text-primary animate-pulse flex items-center gap-2 mt-2">
            <Loader2 className="h-4 w-4 animate-spin" />
            Listening... speak now to tell your story
          </p>
        )}
      </div>

      {/* Photos */}
      <div>
        <Label className="text-lg font-semibold flex items-center gap-2">
          <Upload className="h-5 w-5" /> Photos
        </Label>
        <div className="mt-2">
          <input
            type="file"
            accept="image/*"
            multiple
            onChange={handlePhotoUpload}
            className="hidden"
            id="memory-photo-upload"
          />
          <label htmlFor="memory-photo-upload">
            <Button
              type="button"
              variant="outline"
              className="w-full h-14 text-lg cursor-pointer"
              disabled={uploadingPhoto}
              asChild
            >
              <span>
                {uploadingPhoto ? (
                  <><Loader2 className="h-5 w-5 mr-2 animate-spin" /> Uploading...</>
                ) : (
                  <><Upload className="h-5 w-5 mr-2" /> Upload Photos</>
                )}
              </span>
            </Button>
          </label>
        </div>
        {formData.photos.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-3">
            {formData.photos.map((url, index) => (
              <div key={index} className="relative">
                <img src={url} alt="" className="h-20 w-20 object-cover rounded-lg" />
                <button
                  type="button"
                  onClick={() => removePhoto(index)}
                  className="absolute -top-2 -right-2 bg-destructive text-destructive-foreground rounded-full p-1"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Submit */}
      <Button
        type="submit"
        variant="accessible"
        className="w-full"
        disabled={loading}
      >
        {loading ? (
          <><Loader2 className="h-5 w-5 mr-2 animate-spin" /> Saving...</>
        ) : (
          <><Plus className="h-5 w-5 mr-2" /> Save Memory</>
        )}
      </Button>
    </form>
  );

  // If used as standalone (with onClose prop), render just the form
  if (onClose) {
    return dialogContent;
  }

  // Otherwise render with dialog trigger
  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button variant="accessible" className="gap-2">
          <Plus className="h-5 w-5" />
          Add Memory
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-lg max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-2xl font-display flex items-center gap-2">
            <Calendar className="h-6 w-6 text-primary" />
            Add New Memory
          </DialogTitle>
        </DialogHeader>
        {dialogContent}
      </DialogContent>
    </Dialog>
  );
};

export default AddMemoryForm;
